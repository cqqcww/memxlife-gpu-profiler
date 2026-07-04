from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def randn(shape, generator, scale=0.02):
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    vocab = int(config["vocab_size"])
    intermediate = int(config["intermediate_size"])
    head_dim = hidden // heads

    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)

    state = {
        "tok_embeddings.weight": randn((vocab, hidden), gen),
        "norm.weight": torch.ones(hidden, dtype=torch.float32),
        "output.weight": randn((vocab, hidden), gen),
    }
    for idx in range(layers):
        prefix = f"layers.{idx}"
        state[f"{prefix}.attention_norm.weight"] = torch.ones(hidden, dtype=torch.float32)
        state[f"{prefix}.ffn_norm.weight"] = torch.ones(hidden, dtype=torch.float32)
        state[f"{prefix}.attention.wq.weight"] = randn((heads * head_dim, hidden), gen)
        state[f"{prefix}.attention.wk.weight"] = randn((kv_heads * head_dim, hidden), gen)
        state[f"{prefix}.attention.wv.weight"] = randn((kv_heads * head_dim, hidden), gen)
        state[f"{prefix}.attention.wo.weight"] = randn((hidden, hidden), gen)
        state[f"{prefix}.feed_forward.w1.weight"] = randn((intermediate, hidden), gen)
        state[f"{prefix}.feed_forward.w2.weight"] = randn((hidden, intermediate), gen)
        state[f"{prefix}.feed_forward.w3.weight"] = randn((intermediate, hidden), gen)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
