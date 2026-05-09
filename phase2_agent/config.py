from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_local_api_config(repo_root: Path) -> dict[str, str]:
    candidates = [
        repo_root / "api_config.py",
        repo_root / "memxlife-project" / "api_config.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("_phase2_api_config", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            "api_key": getattr(module, "OPENAI_API_KEY", "") or getattr(module, "API_KEY", ""),
            "base_url": getattr(module, "OPENAI_BASE_URL", "") or getattr(module, "BASE_URL", ""),
            "model": getattr(module, "OPENAI_MODEL", "") or getattr(module, "BASE_MODEL", ""),
        }
    return {"api_key": "", "base_url": "", "model": ""}


@dataclass
class LLMSettings:
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-5.4"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


@dataclass
class AgentSettings:
    repo_root: Path
    work_dir: Path
    optimized_path: Path
    history_path: Path
    max_minutes: float = 28.0
    rank: int = 16
    benchmark_sizes: list[int] = field(default_factory=lambda: [3584, 4096, 4608])
    correctness_sizes: list[int] = field(default_factory=lambda: [3584, 4608])
    warmup: int = 6
    benchmark_iters: int = 15
    llm: LLMSettings = field(default_factory=LLMSettings)

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "AgentSettings":
        repo_root = repo_root.resolve()
        local = _load_local_api_config(repo_root)
        api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or local["api_key"]
        base_url = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL") or local["base_url"]
        model = os.environ.get("BASE_MODEL") or os.environ.get("OPENAI_MODEL") or local["model"] or "gpt-5.4"
        work_dir = repo_root / ".phase2_work"
        return cls(
            repo_root=repo_root,
            work_dir=work_dir,
            optimized_path=repo_root / "optimized_lora.cu",
            history_path=work_dir / "history.json",
            llm=LLMSettings(api_key=api_key, base_url=base_url.rstrip("/"), model=model),
        )

