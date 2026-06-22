"""Checkpoint save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path


class CheckpointManager:
    def __init__(self, run_dir: Path, keep_last: int = 2, save_rng: bool = True):
        self.run_dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.save_rng = save_rng

    def save(self, step: int, model, optimizer, scheduler, extra: dict | None = None) -> Path:
        import torch

        path = self.ckpt_dir / f"step_{step:06d}.pt"
        payload = {
            "global_step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "extra": extra or {},
        }
        if self.save_rng:
            payload["rng"] = {"torch": torch.get_rng_state()}
            if torch.cuda.is_available():
                payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)
        self._cleanup()
        (self.ckpt_dir / "latest.txt").write_text(str(path.name), encoding="utf-8")
        return path

    def load(self, path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
        ckpt_path = self._resolve_checkpoint_path(path)
        import torch

        payload = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if self.save_rng:
            self._restore_rng(payload.get("rng"), torch)
        payload["_loaded_checkpoint_path"] = str(ckpt_path)
        return payload

    def write_metadata(self, data: dict) -> None:
        (self.ckpt_dir / "metadata.json").write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _cleanup(self) -> None:
        if self.keep_last <= 0:
            return
        checkpoints = sorted(self.ckpt_dir.glob("step_*.pt"))
        for path in checkpoints[: -self.keep_last]:
            path.unlink(missing_ok=True)

    def _resolve_checkpoint_path(self, path: str | Path) -> Path:
        ckpt_path = Path(path)
        if ckpt_path.is_dir():
            latest_path = ckpt_path / "latest.txt"
            if not latest_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint directory {ckpt_path} does not contain latest.txt"
                )
            latest = latest_path.read_text(encoding="utf-8").strip()
            if not latest:
                raise FileNotFoundError(
                    f"Checkpoint directory {ckpt_path} has an empty latest.txt"
                )
            ckpt_path = ckpt_path / latest
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint path is not a file: {ckpt_path}")
        return ckpt_path

    def _restore_rng(self, rng_payload, torch) -> None:
        if not rng_payload:
            return
        if rng_payload.get("torch") is not None:
            torch.set_rng_state(rng_payload["torch"].cpu())
        cuda_states = rng_payload.get("cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
