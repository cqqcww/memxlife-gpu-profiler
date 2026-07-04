from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


def load_engine_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(device: str, fn):
    sync_if_needed(device)
    started = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    return time.perf_counter() - started, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weight-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    module = load_engine_module(Path(args.engine))
    vocab = int(config["vocab_size"])

    same_ids = list(range(args.batch_size))
    same_prompts = [
        torch.arange(i + 1, i + 1 + args.prompt_len, dtype=torch.long).remainder(vocab)
        for i in same_ids
    ]
    mixed_lengths = [max(1, args.prompt_len - 4 * i) for i in range(args.batch_size)]
    mixed_ids = list(range(1000, 1000 + args.batch_size))
    mixed_prompts = [
        torch.arange(i + 1, i + 1 + length, dtype=torch.long).remainder(vocab)
        for i, length in enumerate(mixed_lengths)
    ]

    same_engine = module.create_engine(config, args.weight_dir, device)
    mixed_engine = module.create_engine(config, args.weight_dir, device)

    def run_same_decode():
        for step in range(args.decode_steps):
            tokens = torch.tensor([(step + i + 17) % vocab for i in same_ids], dtype=torch.long)
            same_engine.decode(same_ids, tokens)

    def run_mixed_decode():
        for step in range(args.decode_steps):
            tokens = torch.tensor([(step + i + 33) % vocab for i in range(args.batch_size)], dtype=torch.long)
            mixed_engine.decode(mixed_ids, tokens)

    same_prefill_samples = []
    same_decode_samples = []
    mixed_prefill_samples = []
    mixed_decode_samples = []
    rounds = args.warmup_rounds + args.repeats
    for round_idx in range(rounds):
        record = round_idx >= args.warmup_rounds

        same_prefill_time, _ = timed(device, lambda: same_engine.prefill(same_ids, same_prompts))
        same_engine.prefill(same_ids, same_prompts)
        same_decode_time, _ = timed(device, run_same_decode)

        mixed_prefill_time, _ = timed(device, lambda: mixed_engine.prefill(mixed_ids, mixed_prompts))
        mixed_engine.prefill(mixed_ids, mixed_prompts)
        mixed_decode_time, _ = timed(device, run_mixed_decode)

        if record:
            same_prefill_samples.append(same_prefill_time)
            same_decode_samples.append(same_decode_time)
            mixed_prefill_samples.append(mixed_prefill_time)
            mixed_decode_samples.append(mixed_decode_time)

    same_prefill_time = statistics.median(same_prefill_samples)
    same_decode_time = statistics.median(same_decode_samples)
    mixed_prefill_time = statistics.median(mixed_prefill_samples)
    mixed_decode_time = statistics.median(mixed_decode_samples)

    print(
        json.dumps(
            {
                "same_prefill_seconds": same_prefill_time,
                "same_decode_seconds": same_decode_time,
                "mixed_prefill_seconds": mixed_prefill_time,
                "mixed_decode_seconds": mixed_decode_time,
                "same_prefill_tokens_per_second": args.batch_size * args.prompt_len / same_prefill_time,
                "same_decode_tokens_per_second": args.batch_size * args.decode_steps / same_decode_time,
                "mixed_prefill_tokens_per_second": sum(mixed_lengths) / mixed_prefill_time,
                "mixed_decode_tokens_per_second": args.batch_size * args.decode_steps / mixed_decode_time,
                "samples": {
                    "same_prefill_seconds": same_prefill_samples,
                    "same_decode_seconds": same_decode_samples,
                    "mixed_prefill_seconds": mixed_prefill_samples,
                    "mixed_decode_seconds": mixed_decode_samples,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
