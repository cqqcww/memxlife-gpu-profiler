from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

from phase2_agent.candidate_space import CandidateConfig, CandidateResult, write_candidate_source
from phase2_agent.codegen import render_candidate
from phase2_agent.config import AgentSettings


class LocalHarness:
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self._torch = None
        self._load = None

    def is_ready(self) -> tuple[bool, str]:
        try:
            import torch
            from torch.utils.cpp_extension import load
        except Exception as exc:  # pragma: no cover - import failure is environment-dependent
            return False, f"PyTorch extension toolchain unavailable: {exc}"
        if not torch.cuda.is_available():
            return False, "CUDA is not available in the current environment"
        self._torch = torch
        self._load = load
        return True, ""

    def evaluate(self, candidate: CandidateConfig) -> CandidateResult:
        ready, reason = self.is_ready()
        result = CandidateResult(candidate=candidate)
        if not ready:
            result.error = reason
            return result

        torch = self._torch
        assert torch is not None
        load = self._load
        assert load is not None

        code = render_candidate(candidate)
        candidate_id = candidate.stable_id()
        source_path = self.settings.work_dir / "candidates" / f"{candidate.slug}-{candidate_id}.cu"
        build_dir = self.settings.work_dir / "builds" / f"{candidate.slug}-{candidate_id}"
        write_candidate_source(source_path, code)
        result.source_path = str(source_path)

        module_name = "optimized_lora_" + hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
        result.module_name = module_name

        try:
            module = load(
                name=module_name,
                sources=[str(source_path)],
                verbose=False,
                with_cuda=True,
                extra_cuda_cflags=["-O3"],
                build_directory=str(build_dir),
            )
            result.compile_ok = True
        except Exception as exc:
            result.error = f"compile failed: {exc}"
            return result

        correctness = []
        for size in self.settings.correctness_sizes:
            passed, max_abs_err, rel_l2_err = self._check_correctness(module, size)
            correctness.append((passed, max_abs_err, rel_l2_err))
            if not passed:
                result.correct = False
                result.max_abs_err = max_abs_err
                result.rel_l2_err = rel_l2_err
                result.error = f"correctness failed at size {size}"
                return result

        result.correct = True
        result.max_abs_err = max(x[1] for x in correctness)
        result.rel_l2_err = max(x[2] for x in correctness)

        student_times = []
        torch_times = []
        for size in self.settings.benchmark_sizes:
            student_ms, torch_ms = self._benchmark_pair(module, size)
            student_times.append(student_ms)
            torch_times.append(torch_ms)

        result.student_ms = statistics.median(student_times)
        result.torch_ms = statistics.median(torch_times)
        if result.student_ms and result.student_ms > 0:
            result.speedup = result.torch_ms / result.student_ms
        return result

    def _make_inputs(self, size: int):
        torch = self._torch
        assert torch is not None
        rank = self.settings.rank
        gen = torch.Generator(device="cuda")
        gen.manual_seed(size + rank)
        return (
            torch.randn((size, size), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
            torch.randn((size, size), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
            torch.randn((size, rank), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
            torch.randn((size, rank), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
        )

    def _reference_impl(self, W, X, A, B):
        torch = self._torch
        assert torch is not None
        with torch.no_grad():
            return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)

    def _check_correctness(self, module, size: int):
        torch = self._torch
        assert torch is not None
        W, X, A, B = self._make_inputs(size)
        with torch.no_grad():
            y_student = module.forward(W, X, A, B)
            y_ref = self._reference_impl(W, X, A, B)
        diff = (y_student - y_ref).float()
        max_abs_err = diff.abs().max().item()
        rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
        passed = torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4)
        return passed, max_abs_err, rel_l2_err

    def _benchmark_pair(self, module, size: int):
        torch = self._torch
        assert torch is not None
        W, X, A, B = self._make_inputs(size)
        student_ms = self._benchmark(lambda: module.forward(W, X, A, B))
        torch_ms = self._benchmark(lambda: self._reference_impl(W, X, A, B))
        return student_ms, torch_ms

    def _benchmark(self, fn):
        torch = self._torch
        assert torch is not None
        for _ in range(self.settings.warmup):
            _ = fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(self.settings.benchmark_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        samples.sort()
        return samples[len(samples) // 2]

    def write_bootstrap_candidate(self, candidate: CandidateConfig) -> Path:
        code = render_candidate(candidate)
        self.settings.optimized_path.write_text(code, encoding="utf-8")
        return self.settings.optimized_path

    def persist_history(self, records: list[dict]) -> None:
        self.settings.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": records,
        }
        self.settings.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

