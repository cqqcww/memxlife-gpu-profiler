from __future__ import annotations

import argparse
import importlib.util
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weight-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode-steps", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    module = load_engine_module(Path(args.engine))
    engine = module.create_engine(config, args.weight_dir, device)

    vocab = int(config["vocab_size"])
    prompt_lengths = [4, 8, 12, 16, 20, 24, 28, 32]
    prompts = [
        torch.arange(i + 1, i + 1 + length, dtype=torch.long).remainder(vocab)
        for i, length in enumerate(prompt_lengths)
    ]
    request_ids = list(range(len(prompts)))

    sync_if_needed(device)
    started = time.perf_counter()
    engine.prefill(request_ids, prompts)
    total_tokens = sum(prompt_lengths)
    for step in range(args.decode_steps):
        tokens = torch.tensor([(step + i + 33) % vocab for i in request_ids], dtype=torch.long)
        engine.decode(request_ids, tokens)
        total_tokens += len(request_ids)
    sync_if_needed(device)
    elapsed = time.perf_counter() - started
    print(json.dumps({"elapsed_seconds": elapsed, "tokens": total_tokens, "tokens_per_second": total_tokens / elapsed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
