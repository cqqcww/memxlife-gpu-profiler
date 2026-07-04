from __future__ import annotations

import argparse
import importlib.util
import json
import random
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


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.allclose(actual, expected, atol=1e-2, rtol=1e-2):
        max_abs = (actual - expected).abs().max().item()
        raise AssertionError(f"{label} failed: max_abs={max_abs}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weight-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    module = load_engine_module(Path(args.engine))
    engine = module.create_engine(config, args.weight_dir, device)
    vocab = int(config["vocab_size"])

    active: dict[int, torch.Tensor] = {}
    next_request_id = 1000
    for step in range(40):
        if len(active) < 6 and (not active or rng.random() < 0.45):
            batch_size = rng.randint(1, 3)
            request_ids = []
            prompts = []
            for _ in range(batch_size):
                request_id = next_request_id
                next_request_id += 1
                prompt_len = rng.randint(1, 12)
                prompt = torch.tensor([rng.randrange(vocab) for _ in range(prompt_len)], dtype=torch.long)
                request_ids.append(request_id)
                prompts.append(prompt)
                active[request_id] = prompt.clone()

            logits = engine.prefill(request_ids, prompts)
            for row, request_id in enumerate(request_ids):
                assert_close(logits[row], engine._forward_full(active[request_id].to(device)), f"prefill step {step} id {request_id}")
            continue

        decode_ids = rng.sample(list(active), rng.randint(1, min(4, len(active))))
        token_ids = torch.tensor([rng.randrange(vocab) for _ in decode_ids], dtype=torch.long)
        logits = engine.decode(decode_ids, token_ids)
        for row, request_id in enumerate(decode_ids):
            active[request_id] = torch.cat([active[request_id], token_ids[row : row + 1].cpu()])
            assert_close(logits[row], engine._forward_full(active[request_id].to(device)), f"decode step {step} id {request_id}")

        if active and rng.random() < 0.25:
            remove_count = rng.randint(1, min(2, len(active)))
            remove_ids = rng.sample(list(active), remove_count)
            engine.remove(remove_ids)
            for request_id in remove_ids:
                active.pop(request_id, None)

    print("stress correctness passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
