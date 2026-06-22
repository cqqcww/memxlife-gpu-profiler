#!/usr/bin/env python3
"""Phase 4 mini training framework entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from training_framework.config import load_config
from training_framework.config_merge import load_composed_config
from training_framework.trainer import Trainer


def main() -> int:
    # The course image can have an older ONNX/protobuf pair. Torch imports ONNX
    # through Dynamo even when we do not use ONNX directly, so set the official
    # protobuf compatibility escape hatch before torch is imported downstream.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to a complete YAML/JSON config")
    parser.add_argument("--base", help="Base config used with profile composition")
    parser.add_argument("--model-profile", help="Optional model profile YAML")
    parser.add_argument("--data-profile", help="Optional data profile YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, e.g. trainer.max_steps=10. May be repeated.",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build tokenizer/model, write preflight artifacts, and exit before optimizer/training.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    if args.config and (args.base or args.model_profile or args.data_profile or args.override):
        parser.error("--config is mutually exclusive with --base/--model-profile/--data-profile/--override")
    if not args.config and not args.base:
        parser.error("one of --config or --base is required")

    if args.config:
        config = load_config(args.config)
    else:
        config = load_composed_config(
            args.base,
            model_profile=args.model_profile,
            data_profile=args.data_profile,
            overrides=args.override,
            project_root=project_root,
        )
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0

    trainer = Trainer(config, project_root=project_root)
    summary = trainer.preflight_only() if args.preflight_only else trainer.fit()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
