"""LLM client abstraction — supports Anthropic Claude (primary) and OpenAI (fallback)."""

from __future__ import annotations

import json
import logging
from typing import Any

from config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client with automatic fallback."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._anthropic = None
        self._openai = None

    def _get_anthropic(self):
        if self._anthropic is None:
            try:
                import anthropic
                self._anthropic = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            except (ImportError, Exception) as e:
                logger.warning("Anthropic client unavailable: %s", e)
        return self._anthropic

    def _get_openai(self):
        if self._openai is None:
            try:
                import openai
                self._openai = openai.OpenAI(api_key=self.config.openai_api_key)
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
        """Send a completion request. Tries primary provider, falls back if needed."""
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

        # Try primary
        try:
            if provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt, model, temp, max_tok)
            else:
                return self._call_openai(system_prompt, user_prompt, model, temp, max_tok, response_format)
        except Exception as e:
            logger.warning("Primary LLM call failed (%s): %s", provider, e)

        # Fallback
        fallback_model = self.config.fallback_model
        logger.info("Falling back to %s / %s", self.config.fallback_provider, fallback_model)
        try:
            if self.config.fallback_provider == "openai":
                return self._call_openai(system_prompt, user_prompt, fallback_model, temp, max_tok, response_format)
            else:
                return self._call_anthropic(system_prompt, user_prompt, fallback_model, temp, max_tok)
        except Exception as e2:
            raise RuntimeError(f"Both primary and fallback LLM calls failed: {e2}") from e2

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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
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
