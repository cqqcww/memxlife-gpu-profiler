from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Callable

from phase2_agent.candidate_space import CandidateConfig, CandidateResult, write_candidate_source
from phase2_agent.codegen import render_candidate
from phase2_agent.config import AgentSettings
from phase2_agent.tracing import TraceLogger


class LocalHarness:
    def __init__(self, settings: AgentSettings, tracer: TraceLogger):
        self.settings = settings
        self.tracer = tracer
        self._torch = None
        self._load = None

    def is_ready(self) -> tuple[bool, str]:
        try:
            import torch
            from torch.utils.cpp_extension import load
        except Exception as exc:  # pragma: no cover - import failure is environment-dependent
            self.tracer.log("environment_not_ready", reason=f"PyTorch extension toolchain unavailable: {exc}")
            return False, f"PyTorch extension toolchain unavailable: {exc}"
        if not torch.cuda.is_available():
            self.tracer.log("environment_not_ready", reason="CUDA is not available in the current environment")
            return False, "CUDA is not available in the current environment"
        self._torch = torch
        self._load = load
        self.tracer.log("environment_ready", torch_version=getattr(torch, "__version__", "unknown"))
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
        build_dir.mkdir(parents=True, exist_ok=True)
        write_candidate_source(source_path, code)
        result.source_path = str(source_path)
        self.tracer.log(
            "candidate_source_written",
            candidate=result.candidate.to_dict() if hasattr(result.candidate, "to_dict") else result.to_dict()["candidate"],
            source_path=str(source_path),
            source_lines=len(code.splitlines()),
            source_sha1=hashlib.sha1(code.encode("utf-8")).hexdigest(),
        )

        module_name = "optimized_lora_" + hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
        result.module_name = module_name

        try:
            compile_started_at = time.time()
            self.tracer.log(
                "compile_started",
                candidate_id=candidate_id,
                module_name=module_name,
                build_directory=str(build_dir),
                extra_cuda_cflags=["-O3"],
            )
            module = load(
                name=module_name,
                sources=[str(source_path)],
                verbose=False,
                with_cuda=True,
                extra_cuda_cflags=["-O3"],
                build_directory=str(build_dir),
            )
            result.compile_ok = True
            result.compile_seconds = time.time() - compile_started_at
            self.tracer.log(
                "compile_succeeded",
                candidate_id=candidate_id,
                module_name=module_name,
                compile_seconds=result.compile_seconds,
            )
        except Exception as exc:
            result.error = f"compile failed: {exc}"
            self.tracer.log("compile_failed", candidate_id=candidate_id, error=result.error)
            return result

        correctness = []
        for size in self.settings.correctness_sizes:
            passed, max_abs_err, rel_l2_err = self._check_correctness(module, size)
            correctness.append((passed, max_abs_err, rel_l2_err))
            self.tracer.log(
                "correctness_checked",
                candidate_id=candidate_id,
                size=size,
                passed=passed,
                max_abs_err=max_abs_err,
                rel_l2_err=rel_l2_err,
            )
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
        repeated_times = []
        combined_speedups = []
        debug_stats_by_size = {}
        for size in self.settings.benchmark_sizes:
            student_ms, torch_ms, cached_ms, repeat_torch_ms, combined_speedup, debug_stats = self._benchmark_pair(module, size)
            student_times.append(student_ms)
            torch_times.append(torch_ms)
            repeated_times.append(cached_ms)
            combined_speedups.append(combined_speedup)
            debug_stats_by_size[str(size)] = debug_stats
            self.tracer.log(
                "benchmark_completed",
                candidate_id=candidate_id,
                size=size,
                student_ms=student_ms,
                torch_ms=torch_ms,
                cached_repeat_ms=cached_ms,
                repeat_torch_ms=repeat_torch_ms,
                varying_x_speedup=(torch_ms / student_ms) if student_ms > 0 else 0.0,
                repeated_x_speedup=(repeat_torch_ms / cached_ms) if cached_ms > 0 else 0.0,
                speedup=combined_speedup,
                debug_stats=debug_stats,
            )

        result.student_ms = statistics.median(student_times)
        result.torch_ms = statistics.median(torch_times)
        result.cached_repeat_ms = statistics.median(repeated_times) if repeated_times else None
        if combined_speedups:
            result.speedup = statistics.median(combined_speedups)
        result.debug_stats = debug_stats_by_size
        self.tracer.log(
            "candidate_evaluation_finished",
            candidate_id=candidate_id,
            compile_ok=result.compile_ok,
            correct=result.correct,
            max_abs_err=result.max_abs_err,
            rel_l2_err=result.rel_l2_err,
            compile_seconds=result.compile_seconds,
            median_student_ms=result.student_ms,
            median_torch_ms=result.torch_ms,
            median_cached_repeat_ms=result.cached_repeat_ms,
            speedup=result.speedup,
            debug_stats=result.debug_stats,
        )
        return result

    def _make_inputs(self, size: int, salt: int = 0):
        torch = self._torch
        assert torch is not None
        rank = self.settings.rank
        gen = torch.Generator(device="cuda")
        gen.manual_seed((size + rank) * 1009 + salt)
        return (
            torch.randn((size, size), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
            torch.randn((size, size), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
            torch.randn((size, rank), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
            torch.randn((size, rank), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
        )

    def _make_activation(self, size: int, salt: int = 0):
        torch = self._torch
        assert torch is not None
        gen = torch.Generator(device="cuda")
        gen.manual_seed((size + self.settings.rank) * 2027 + salt)
        return torch.randn((size, size), device="cuda", dtype=torch.float32, generator=gen).contiguous()

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
        W, X, A, B = self._make_inputs(size, salt=0)
        varying_xs = [self._make_activation(size, salt=i + 1) for i in range(3)]

        self._reset_debug_stats(module)
        varying_student_ms = self._benchmark_sequence(
            [lambda X_i=X_i: module.forward(W, X_i, A, B) for X_i in varying_xs]
        )
        varying_stats = self._read_debug_stats(module)
        varying_torch_ms = self._benchmark_sequence(
            [lambda X_i=X_i: self._reference_impl(W, X_i, A, B) for X_i in varying_xs]
        )
        self._reset_debug_stats(module)
        repeated_student_ms = self._benchmark(lambda: module.forward(W, X, A, B))
        repeated_stats = self._read_debug_stats(module)
        repeated_torch_ms = self._benchmark(lambda: self._reference_impl(W, X, A, B))

        varying_speedup = (varying_torch_ms / varying_student_ms) if varying_student_ms > 0 else 0.0
        repeated_speedup = (repeated_torch_ms / repeated_student_ms) if repeated_student_ms > 0 else 0.0
        combined_speedup = self._harmonic_mean(varying_speedup, repeated_speedup)
        return (
            varying_student_ms,
            varying_torch_ms,
            repeated_student_ms,
            repeated_torch_ms,
            combined_speedup,
            {
                "varying": varying_stats,
                "repeated": repeated_stats,
            },
        )

    def _reset_debug_stats(self, module) -> None:
        reset = getattr(module, "reset_debug_stats", None)
        if reset is None:
            return
        try:
            reset()
        except Exception as exc:
            self.tracer.log("debug_stats_reset_failed", error=str(exc))

    def _read_debug_stats(self, module) -> dict:
        reader = getattr(module, "get_debug_stats", None)
        if reader is None:
            return {}
        try:
            raw = reader()
        except Exception as exc:
            self.tracer.log("debug_stats_read_failed", error=str(exc))
            return {}
        return self._normalize_debug_stats(raw)

    def _normalize_debug_stats(self, raw) -> dict:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            items = raw.items()
        elif hasattr(raw, "items"):
            items = raw.items()
        else:
            return {"value": str(raw)}
        normalized = {}
        for key, value in items:
            normalized[str(key)] = self._normalize_debug_scalar(value)
        return normalized

    @staticmethod
    def _normalize_debug_scalar(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        try:
            if hasattr(value, "item"):
                item = value.item()
                if isinstance(item, (bool, int, float)):
                    return item
        except Exception:
            pass
        return str(value)

    def _benchmark(self, fn: Callable[[], object]):
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

    def _benchmark_sequence(self, fns: list[Callable[[], object]]):
        torch = self._torch
        assert torch is not None
        if not fns:
            return 0.0
        for _ in range(self.settings.warmup):
            for fn in fns:
                _ = fn()
        torch.cuda.synchronize()
        samples = []
        for idx in range(self.settings.benchmark_iters):
            fn = fns[idx % len(fns)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        samples.sort()
        return samples[len(samples) // 2]

    @staticmethod
    def _harmonic_mean(lhs: float, rhs: float) -> float:
        if lhs <= 0 or rhs <= 0:
            return 0.0
        return 2.0 / ((1.0 / lhs) + (1.0 / rhs))

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
