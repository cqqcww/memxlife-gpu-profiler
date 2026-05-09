from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CandidateConfig:
    strategy: str
    main_backend: str
    low_rank_backend: str
    accumulation_order: str
    allow_tf32: bool
    notes: str = ""

    @property
    def slug(self) -> str:
        tf32 = "tf32" if self.allow_tf32 else "fp32"
        return f"{self.strategy}-{self.main_backend}-{self.low_rank_backend}-{self.accumulation_order}-{tf32}"

    def stable_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


@dataclass
class CandidateResult:
    candidate: CandidateConfig
    compile_ok: bool = False
    correct: bool = False
    max_abs_err: float | None = None
    rel_l2_err: float | None = None
    student_ms: float | None = None
    torch_ms: float | None = None
    speedup: float = 0.0
    error: str = ""
    module_name: str = ""
    source_path: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["candidate"] = asdict(self.candidate)
        return data


def heuristic_candidates() -> list[CandidateConfig]:
    return [
        CandidateConfig(
            strategy="aten",
            main_backend="aten_matmul",
            low_rank_backend="aten_addmm",
            accumulation_order="wx_then_lora",
            allow_tf32=True,
            notes="Bootstrap candidate using ATen ops for guaranteed correctness.",
        ),
        CandidateConfig(
            strategy="cublas",
            main_backend="sgemm",
            low_rank_backend="sgemm",
            accumulation_order="wx_then_lora",
            allow_tf32=False,
            notes="Deterministic cuBLAS SGEMM path.",
        ),
        CandidateConfig(
            strategy="cublas",
            main_backend="gemm_ex_tf32",
            low_rank_backend="sgemm",
            accumulation_order="wx_then_lora",
            allow_tf32=True,
            notes="TF32 for dominant GEMM, FP32 for low-rank update.",
        ),
        CandidateConfig(
            strategy="cublas",
            main_backend="gemm_ex_tf32",
            low_rank_backend="gemm_ex_tf32",
            accumulation_order="wx_then_lora",
            allow_tf32=True,
            notes="Aggressive TF32 for all GEMMs.",
        ),
        CandidateConfig(
            strategy="cublas",
            main_backend="gemm_ex_tf32",
            low_rank_backend="sgemm",
            accumulation_order="lora_then_wx",
            allow_tf32=True,
            notes="Same kernels with reversed accumulation order.",
        ),
    ]


def write_candidate_source(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")

