"""Global configuration for the GPU profiling agent system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM provider and model configuration."""
    # Primary provider
    provider: str = "anthropic"  # "anthropic" or "openai"
    # Per-agent model selection
    planner_model: str = "claude-sonnet-4-20250514"
    codegen_model: str = "claude-opus-4-20250514"
    analyzer_model: str = "claude-opus-4-20250514"
    # Fallback
    fallback_provider: str = "openai"
    fallback_model: str = "gpt-4o"
    # API keys from env
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Token budgets
    max_input_tokens: int = 8000
    max_output_tokens: int = 4096
    temperature: float = 0.2

    def __post_init__(self):
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")


@dataclass
class RunConfig:
    """Runtime configuration."""
    # Probe settings
    max_retries_per_metric: int = 3
    max_compile_retries: int = 3
    confidence_threshold: float = 0.7
    # Execution
    compile_timeout_sec: int = 60
    execute_timeout_sec: int = 120
    ncu_timeout_sec: int = 180
    # Paths
    output_dir: str = "runs"
    # Mock mode for local dev without GPU
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
