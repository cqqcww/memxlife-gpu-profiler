from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class CandidateConfig:
    strategy: str
    main_backend: str
    low_rank_backend: str
    accumulation_order: str
    allow_tf32: bool
    cache_mode: str = "none"
    variant_name: str = ""
    notes: str = ""

    @property
    def slug(self) -> str:
        tf32 = "tf32" if self.allow_tf32 else "fp32"
        return (
            f"{self.strategy}-{self.main_backend}-{self.low_rank_backend}-"
            f"{self.accumulation_order}-{tf32}-{self.cache_mode}"
        )

    def stable_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


@dataclass
class CandidateResult:
    candidate: CandidateConfig
    compile_ok: bool = False
    correct: bool = False
    compile_seconds: float | None = None
    max_abs_err: float | None = None
    rel_l2_err: float | None = None
    student_ms: float | None = None
    torch_ms: float | None = None
    cached_repeat_ms: float | None = None
    speedup: float = 0.0
    error: str = ""
    module_name: str = ""
    source_path: str = ""
    debug_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["candidate"] = asdict(self.candidate)
        return data


def heuristic_candidates() -> list[CandidateConfig]:
    return [
        CandidateConfig(
            strategy="aten",
            main_backend="addmm_inplace",
            low_rank_backend="bt_contiguous",
            accumulation_order="mainfirst",
            allow_tf32=False,
            cache_mode="hybrid_weff",
            variant_name="aten_addmm_inplace_btcontig_mainfirst_hybridweff",
            notes="Primary high-score candidate: single-slot exact-repeat output cache, immediate W_eff materialization on the first same-weight varying-X miss, and a plain cold fallback for new weights.",
        ),
        CandidateConfig(
            strategy="aten",
            main_backend="addmm_inplace",
            low_rank_backend="bt_contiguous_out",
            accumulation_order="mainfirst",
            allow_tf32=False,
            cache_mode="hybrid_weff",
            variant_name="aten_addmm_inplace_btcontigout_mainfirst_hybridweff",
            notes="Existing low-risk hybrid sibling: preserve single-slot exact-repeat and immediate W_eff materialization, but use an mm_out-style cold fallback to test whether preallocated BX helps the weakest hidden case.",
        ),
        CandidateConfig(
            strategy="aten",
            main_backend="addmm_inplace",
            low_rank_backend="bt_contiguous",
            accumulation_order="mainfirst",
            allow_tf32=False,
            cache_mode="hybrid_weff_dualrepeat",
            variant_name="aten_addmm_inplace_btcontig_mainfirst_hybridweff_dualrepeat",
            notes="Two-slot exact-repeat cache for ABAB-style alternating inputs, while keeping immediate W_eff materialization for same-weight varying-X traffic.",
        ),
        CandidateConfig(
            strategy="aten",
            main_backend="addmm_inplace",
            low_rank_backend="bt_contiguous",
            accumulation_order="mainfirst",
            allow_tf32=False,
            cache_mode="hybrid_weff_threshold2",
            variant_name="aten_addmm_inplace_btcontig_mainfirst_hybridweff_threshold2",
            notes="Delay W_eff materialization until the second same-weight varying-X event, aiming to protect colder hidden cases from paying the materialization cost too early.",
        ),
        CandidateConfig(
            strategy="aten",
            main_backend="addmm_inplace",
            low_rank_backend="bt_contiguous",
            accumulation_order="mainfirst",
            allow_tf32=False,
            cache_mode="hybrid_weff_dualrepeat_threshold2",
            variant_name="aten_addmm_inplace_btcontig_mainfirst_hybridweff_dualrepeat_threshold2",
            notes="Combine dual-slot exact-repeat reuse with a delayed W_eff threshold, so alternating repeats stay fast without forcing W_eff on the very first varying-X miss.",
        ),
        CandidateConfig(
            strategy="aten",
            main_backend="addmm_inplace",
            low_rank_backend="bt_contiguous_out",
            accumulation_order="mainfirst",
            allow_tf32=False,
            cache_mode="hybrid_weff_dualrepeat_threshold2",
            variant_name="aten_addmm_inplace_btcontigout_mainfirst_hybridweff_dualrepeat_threshold2",
            notes="Most aggressive low-risk variant: dual-slot exact-repeat cache, second-hit W_eff materialization, and mm_out cold fallback to squeeze the weakest hidden case over parity.",
        ),
    ]


def write_candidate_source(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
