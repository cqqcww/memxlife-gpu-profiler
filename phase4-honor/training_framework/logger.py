"""Console, JSONL, and TensorBoard logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: Path, enable_tensorboard: bool, enable_jsonl: bool, enable_console: bool):
        self.run_dir = run_dir
        self.enable_console = enable_console
        self.jsonl_path = run_dir / "events.jsonl"
        self.jsonl_file = None
        if enable_jsonl:
            self.jsonl_file = self.jsonl_path.open("a", encoding="utf-8")
        self.writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(run_dir / "tb"))
            except Exception as exc:
                if enable_console:
                    print(f"[logger] TensorBoard disabled: {exc}")

    def log(self, event: str, step: int, metrics: dict[str, Any]) -> None:
        record = {
            "time": time.time(),
            "event": event,
            "step": step,
            "metrics": metrics,
        }
        if self.jsonl_file:
            self.jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
            self.jsonl_file.flush()
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{event}/{key}", value, step)
        if self.enable_console:
            compact = " ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            print(f"[{event}] step={step} {compact}", flush=True)

    def close(self) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()
        if self.jsonl_file:
            self.jsonl_file.close()
