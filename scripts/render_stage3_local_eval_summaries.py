from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_json_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    buf: list[str] = []
    depth = 0
    capturing = False
    for line in text.splitlines():
        opens = line.count("{")
        closes = line.count("}")
        if not capturing and opens:
            capturing = True
        if capturing:
            buf.append(line)
            depth += opens
            depth -= closes
            if depth == 0:
                raw = "\n".join(buf).strip()
                if raw:
                    blocks.append(json.loads(raw))
                buf = []
                capturing = False
    return blocks


def render_public_summary(log_path: Path, summary_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8")
    statuses = [line.strip() for line in text.splitlines() if "passed" in line.lower()]
    blocks = extract_json_blocks(text)

    throughput = blocks[0] if len(blocks) >= 1 else None
    mixed = blocks[1] if len(blocks) >= 2 else None

    lines: list[str] = [
        "# Remote Public Eval Summary",
        "",
        f"Source log: [{log_path.name}]({log_path.resolve()})",
        "",
        "## Status",
        "",
    ]

    if statuses:
        lines.extend([f"- {line}" for line in statuses])
    else:
        lines.append("- No explicit `passed` lines found in the log.")

    lines.extend(["", "## Throughput", ""])

    if throughput is not None:
        lines.extend(
            [
                "### Benchmark throughput",
                "",
                "```json",
                json.dumps(throughput, indent=2),
                "```",
                "",
            ]
        )

    if mixed is not None:
        lines.extend(
            [
                "### Benchmark mixed",
                "",
                "```json",
                json.dumps(mixed, indent=2),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.",
            "- This is the local fallback path when the official `outputs3` publishing chain is unstable.",
        ]
    )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_breakdown_summary(json_path: Path, summary_path: Path) -> None:
    text = json_path.read_text(encoding="utf-8")
    blocks = extract_json_blocks(text)
    payload = None
    if blocks:
        payload = blocks[-1]
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        summary_path.write_text(
            "\n".join(
                [
                    "# Remote Breakdown Summary",
                    "",
                    f"Source log: [{json_path.name}]({json_path.resolve()})",
                    "",
                    "## Status",
                    "",
                    "- No complete JSON payload was recovered from this breakdown log.",
                    "- This usually means the remote benchmark command or SSH session died mid-run.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return

    key_metrics = {
        "same_prefill_tokens_per_second": payload["same_prefill_tokens_per_second"],
        "same_decode_tokens_per_second": payload["same_decode_tokens_per_second"],
        "mixed_prefill_tokens_per_second": payload["mixed_prefill_tokens_per_second"],
        "mixed_decode_tokens_per_second": payload["mixed_decode_tokens_per_second"],
    }

    lines = [
        "# Remote Breakdown Summary",
        "",
        f"Source log: [{json_path.name}]({json_path.resolve()})",
        "",
        "## Breakdown",
        "",
        "```json",
        json.dumps(key_metrics, indent=2),
        "```",
        "",
        "## Read",
        "",
        "- `same_prefill` reflects the uniform prefill path.",
        "- `same_decode` reflects the uniform decode path.",
        "- `mixed_prefill` and `mixed_decode` are the most useful local fallback signals for serving-like traces.",
    ]

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-log", required=True)
    parser.add_argument("--public-summary", required=True)
    parser.add_argument("--breakdown-json", required=True)
    parser.add_argument("--breakdown-summary", required=True)
    args = parser.parse_args()

    render_public_summary(Path(args.public_log), Path(args.public_summary))
    render_breakdown_summary(Path(args.breakdown_json), Path(args.breakdown_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
