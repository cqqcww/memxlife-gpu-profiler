#!/usr/bin/env python3
"""MemXLife GPU Profiling Agent — CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from core.orchestrator import Orchestrator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemXLife GPU Hardware Profiling Agent System",
    )
    parser.add_argument(
        "target_spec",
        help="Path to target_spec.json",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="runs",
        help="Output directory for run artifacts (default: runs)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no GPU required)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all supported metrics and exit",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.list_metrics:
        from knowledge.metrics_catalog import list_supported_metrics
        print("Supported metrics:")
        for m in list_supported_metrics():
            print(f"  - {m}")
        return

    # Validate target spec exists
    spec_path = Path(args.target_spec)
    if not spec_path.exists():
        print(f"Error: target_spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    # Build config
    config = Config.from_env()
    if args.mock:
        config.run.mock_mode = True
    config.run.output_dir = args.output_dir

    # Run
    orchestrator = Orchestrator(config)
    results = orchestrator.run(
        target_spec_path=str(spec_path),
        output_dir=args.output_dir,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
