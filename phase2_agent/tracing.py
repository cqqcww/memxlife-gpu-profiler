from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, jsonl_path: Path, summary_path: Path):
        self.jsonl_path = jsonl_path
        self.summary_path = summary_path
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.summary_lines: list[str] = [
            "# Phase 2 Trace Summary",
            "",
            f"- Run id: `{self.run_id}`",
            "",
        ]
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text("", encoding="utf-8")
        self.summary_path.write_text("\n".join(self.summary_lines) + "\n", encoding="utf-8")

    def log(self, event: str, **data: Any) -> None:
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": self.run_id,
            "event": event,
            **data,
        }
        print("[TRACE] " + json.dumps(payload, ensure_ascii=True), flush=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def add_summary(self, title: str, lines: list[str]) -> None:
        self.summary_lines.append(f"## {title}")
        self.summary_lines.extend(lines)
        self.summary_lines.append("")
        self.summary_path.write_text("\n".join(self.summary_lines) + "\n", encoding="utf-8")
