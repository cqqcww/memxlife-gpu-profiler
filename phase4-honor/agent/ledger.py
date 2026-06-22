"""Append-only run ledger."""

from __future__ import annotations

import json
import time
from pathlib import Path


def append_ledger(path: str | Path, record: dict) -> None:
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": time.time(), **record}
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
