"""LLM client abstraction — supports Anthropic Claude via relay/proxy endpoints.

Features:
  - Custom base_url for relay/proxy API endpoints (e.g., yibuapi.com)
  - Configurable timeout for unstable connections
  - Automatic retry with exponential backoff
  - Fallback between providers
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client with relay support, timeout, and retry."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._anthropic = None
        self._openai = None

    def _get_anthropic(self):
        if self._anthropic is None:
            try:
                import anthropic

                kwargs: dict[str, Any] = {
                    "api_key": self.config.anthropic_api_key,
                    "timeout": float(self.config.request_timeout_sec),
                    "max_retries": 0,  # We handle retries ourselves
                }
                if self.config.anthropic_base_url:
                    kwargs["base_url"] = self.config.anthropic_base_url
                    logger.info(
                        "Anthropic client using relay: %s",
                        self.config.anthropic_base_url,
                    )

                self._anthropic = anthropic.Anthropic(**kwargs)
            except (ImportError, Exception) as e:
                logger.warning("Anthropic client unavailable: %s", e)
        return self._anthropic

    def _get_openai(self):
        if self._openai is None:
            try:
                import openai

                kwargs: dict[str, Any] = {
                    "api_key": self.config.openai_api_key,
                    "timeout": float(self.config.request_timeout_sec),
                    "max_retries": 0,
                }
                if self.config.openai_base_url:
                    kwargs["base_url"] = self.config.openai_base_url

                self._openai = openai.OpenAI(**kwargs)
            except (ImportError, Exception) as e:
                logger.warning("OpenAI client unavailable: %s", e)
        return self._openai

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",  # "text" or "json"
    ) -> str:
        """Send a completion request with retry. Tries primary provider, falls back if needed."""
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_output_tokens
        model = model or self.config.codegen_model

        # Determine provider from model name
        if model.startswith("claude") or model.startswith("anthropic"):
            provider = "anthropic"
        elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
            provider = "openai"
        else:
            provider = self.config.provider

        # Try primary with retry
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                if provider == "anthropic":
                    result = self._call_anthropic(system_prompt, user_prompt, model, temp, max_tok)
                else:
                    result = self._call_openai(system_prompt, user_prompt, model, temp, max_tok, response_format)
                if attempt > 0:
                    logger.info("LLM call succeeded on retry %d", attempt)
                return result
            except Exception as e:
                last_error = e
                delay = self.config.retry_delay_sec * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    "LLM call attempt %d/%d failed (%s): %s — retrying in %.1fs",
                    attempt + 1, self.config.max_retries, provider, e, delay,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(delay)

        logger.warning("Primary provider %s exhausted all %d retries", provider, self.config.max_retries)

        # Fallback provider
        fallback_model = self.config.fallback_model
        fallback_provider = self.config.fallback_provider
        if fallback_provider == provider and fallback_model == model:
            # Same provider/model — no point retrying
            raise RuntimeError(
                f"LLM call failed after {self.config.max_retries} retries: {last_error}"
            ) from last_error

        logger.info("Falling back to %s / %s", fallback_provider, fallback_model)
        for attempt in range(self.config.max_retries):
            try:
                if fallback_provider == "openai":
                    return self._call_openai(system_prompt, user_prompt, fallback_model, temp, max_tok, response_format)
                else:
                    return self._call_anthropic(system_prompt, user_prompt, fallback_model, temp, max_tok)
            except Exception as e2:
                delay = self.config.retry_delay_sec * (2 ** attempt)
                logger.warning(
                    "Fallback attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, self.config.max_retries, e2, delay,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(delay)
                last_error = e2

        raise RuntimeError(
            f"Both primary and fallback LLM calls failed after retries: {last_error}"
        ) from last_error

    def _call_anthropic(
        self, system_prompt: str, user_prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_anthropic()
        if client is None:
            raise RuntimeError("Anthropic client not available")
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_openai(
        self, system_prompt: str, user_prompt: str, model: str, temperature: float, max_tokens: int,
        response_format: str = "text",
    ) -> str:
        client = self._get_openai()
        if client is None:
            raise RuntimeError("OpenAI client not available")

        # Some relay providers drop system messages or return content=None
        # when system+user are separate. Merge them into a single user message.
        merged_prompt = f"[System Instructions]\n{system_prompt}\n\n[User Request]\n{user_prompt}"

        # GPT-5.x requires max_completion_tokens; older/relay APIs use max_tokens
        token_key = "max_completion_tokens" if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") else "max_tokens"

        kwargs: dict[str, Any] = {
            "model": model,
            token_key: max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": merged_prompt},
            ],
        }

        # Use with_raw_response to handle relay APIs that return non-standard formats
        try:
            raw_resp = client.chat.completions.with_raw_response.create(**kwargs)
            import json as _json
            body = _json.loads(raw_resp.text)
            content = body["choices"][0]["message"].get("content")
            if content is not None:
                return content
            return ""
        except (AttributeError, KeyError, Exception) as e:
            logger.debug("Raw response parsing failed (%s), trying standard SDK", e)
            # Fallback to standard SDK parsing
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

    def complete_for_agent(
        self,
        agent_role: str,
        system_prompt: str,
        user_prompt: str,
        response_format: str = "text",
    ) -> str:
        """Convenience method that auto-selects model based on agent role."""
        model_map = {
            "planner": self.config.planner_model,
            "codegen": self.config.codegen_model,
            "analyzer": self.config.analyzer_model,
        }
        model = model_map.get(agent_role, self.config.codegen_model)
        return self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            response_format=response_format,
        )

    def test_connection(self) -> dict[str, Any]:
        """Quick connectivity test — useful for verifying relay config."""
        result: dict[str, Any] = {"ok": False, "provider": "", "model": "", "error": ""}
        try:
            resp = self.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with exactly: OK",
                max_tokens=10,
            )
            result["ok"] = True
            result["provider"] = self.config.provider
            result["model"] = self.config.codegen_model
            result["response"] = resp.strip()
        except Exception as e:
            result["error"] = str(e)
        return result
