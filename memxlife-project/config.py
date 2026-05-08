"""Global configuration for the GPU profiling agent system.

Credential loading priority:
  1. Environment variables: API_KEY, BASE_MODEL, BASE_URL (evaluation)
  2. api_config.py (local dev, not submitted)
  3. Hardcoded defaults (no credentials)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_api_config() -> dict:
    """Try to load api_config.py for local development credentials."""
    try:
        import api_config
        return {
            "openai_api_key": getattr(api_config, "OPENAI_API_KEY", ""),
            "openai_base_url": getattr(api_config, "OPENAI_BASE_URL", ""),
            "openai_model": getattr(api_config, "OPENAI_MODEL", ""),
            "anthropic_api_key": getattr(api_config, "ANTHROPIC_API_KEY", ""),
            "anthropic_base_url": getattr(api_config, "ANTHROPIC_BASE_URL", ""),
            "anthropic_model": getattr(api_config, "ANTHROPIC_MODEL", ""),
        }
    except ImportError:
        return {}


@dataclass
class LLMConfig:
    """LLM provider and model configuration.

    Uses OpenAI-compatible API format (required for GPT-5.4 evaluation).
    """
    provider: str = "openai"
    planner_model: str = ""
    codegen_model: str = ""
    analyzer_model: str = ""
    fallback_provider: str = ""
    fallback_model: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    request_timeout_sec: int = 180
    max_retries: int = 2
    retry_delay_sec: float = 1.0
    max_input_tokens: int = 16000
    max_output_tokens: int = 8192
    temperature: float = 0.2

    def __post_init__(self):
        # Priority 1: Environment variables (evaluation environment)
        env_key = os.environ.get("API_KEY", "")
        env_model = os.environ.get("BASE_MODEL", "")
        env_url = os.environ.get("BASE_URL", "")

        if env_key:
            self.provider = "openai"
            self.openai_api_key = env_key
            self.openai_base_url = env_url
            model = env_model or "gpt-5.4"
            self.planner_model = model
            self.codegen_model = model
            self.analyzer_model = model
            self.fallback_provider = "openai"
            self.fallback_model = model
            return

        # Priority 2: api_config.py (local dev)
        ac = _load_api_config()
        if ac.get("openai_api_key"):
            self.provider = "openai"
            self.openai_api_key = ac["openai_api_key"]
            self.openai_base_url = ac.get("openai_base_url", "")
            model = ac.get("openai_model", "gpt-5.4")
            self.planner_model = model
            self.codegen_model = model
            self.analyzer_model = model
            # Anthropic as fallback
            if ac.get("anthropic_api_key"):
                self.fallback_provider = "anthropic"
                self.fallback_model = ac.get("anthropic_model", "claude-sonnet-4-5-20250929")
                self.anthropic_api_key = ac["anthropic_api_key"]
                self.anthropic_base_url = ac.get("anthropic_base_url", "")
            else:
                self.fallback_provider = "openai"
                self.fallback_model = model
            return

        if ac.get("anthropic_api_key"):
            self.provider = "anthropic"
            self.anthropic_api_key = ac["anthropic_api_key"]
            self.anthropic_base_url = ac.get("anthropic_base_url", "")
            model = ac.get("anthropic_model", "claude-sonnet-4-5-20250929")
            self.planner_model = model
            self.codegen_model = model
            self.analyzer_model = model
            self.fallback_provider = "anthropic"
            self.fallback_model = model
            return

        # Priority 3: Legacy env vars
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        if self.openai_api_key:
            self.provider = "openai"
        elif self.anthropic_api_key:
            self.provider = "anthropic"


@dataclass
class RunConfig:
    """Runtime configuration."""
    max_retries_per_metric: int = 2
    max_compile_retries: int = 3
    confidence_threshold: float = 0.7
    compile_timeout_sec: int = 120
    execute_timeout_sec: int = 120
    ncu_timeout_sec: int = 180
    output_dir: str = "runs"
    mock_mode: bool = False

    def __post_init__(self):
        if os.environ.get("MEMXLIFE_MOCK", "").lower() in ("1", "true", "yes"):
            self.mock_mode = True


@dataclass
class Config:
    """Top-level configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_env(cls) -> Config:
        return cls(llm=LLMConfig(), run=RunConfig())
