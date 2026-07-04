from __future__ import annotations

import argparse
import importlib.util
import json
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
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    module = load_engine_module(Path(args.engine))
    engine = module.create_engine(config, args.weight_dir, device)

    seq0 = torch.tensor([1, 5, 9, 7], dtype=torch.long)
    seq1 = torch.tensor([3, 4, 8], dtype=torch.long)
    logits = engine.prefill([100, 101], [seq0, seq1])
    assert logits.shape == (2, config["vocab_size"])
    assert_close(logits[0], engine._forward_full(engine.requests[100].tokens), "prefill request 100")
    assert_close(logits[1], engine._forward_full(engine.requests[101].tokens), "prefill request 101")

    next_tokens = torch.tensor([11, 12], dtype=torch.long)
    decode_logits = engine.decode([100, 101], next_tokens)
    assert decode_logits.shape == (2, config["vocab_size"])
    assert engine.requests[100].tokens.tolist()[-1] == 11
    assert engine.requests[101].tokens.tolist()[-1] == 12
    assert_close(decode_logits[0], engine._forward_full(engine.requests[100].tokens), "decode request 100")
    assert_close(decode_logits[1], engine._forward_full(engine.requests[101].tokens), "decode request 101")

    engine.prefill([102], [torch.tensor([2, 6], dtype=torch.long)])
    engine.remove([101])
    if 101 in engine.requests:
        raise AssertionError("remove did not delete request 101")
    continued = engine.decode([100, 102], torch.tensor([13, 14], dtype=torch.long))
    assert continued.shape == (2, config["vocab_size"])
    assert_close(continued[0], engine._forward_full(engine.requests[100].tokens), "continued request 100")
    assert_close(continued[1], engine._forward_full(engine.requests[102].tokens), "continued request 102")

    engine.prefill(
        [200, 201, 202],
        [
            torch.tensor([7, 8, 9], dtype=torch.long),
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([13, 14, 15], dtype=torch.long),
        ],
    )
    batched = engine.decode([200, 201, 202], torch.tensor([16, 17, 18], dtype=torch.long))
    assert batched.shape == (3, config["vocab_size"])
    for row, request_id in enumerate([200, 201, 202]):
        assert_close(batched[row], engine._forward_full(engine.requests[request_id].tokens), f"batched decode {request_id}")

    tensor_id_logits = engine.prefill(torch.tensor([300], dtype=torch.long), [torch.tensor([4, 5], dtype=torch.long)])
    assert tensor_id_logits.shape == (1, config["vocab_size"])
    tensor_id_decode = engine.decode(torch.tensor([300], dtype=torch.long), torch.tensor([6], dtype=torch.long))
    assert_close(tensor_id_decode[0], engine._forward_full(engine.requests[300].tokens), "tensor request id decode")
    engine.remove(torch.tensor([300], dtype=torch.long))
    if 300 in engine.requests:
        raise AssertionError("tensor request id remove did not delete request 300")
    print("correctness smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
