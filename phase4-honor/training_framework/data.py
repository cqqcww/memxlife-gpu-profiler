"""Dataset loading, tokenization, block caching, and dataloader creation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .config import DataConfig

CACHE_VERSION = 1


class TokenBlockDataset:
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch is required on the remote training server") from exc

        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }


def _read_local_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    chunks = [part.strip() for part in text.split("\n\n") if part.strip()]
    return chunks or [text]


def _read_hf_dataset(config: DataConfig) -> list[str]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets is required for HuggingFace dataset loading") from exc

    kwargs = {}
    if config.dataset_config:
        kwargs["name"] = config.dataset_config
    ds = load_dataset(config.dataset_name, **kwargs, split=config.dataset_split)
    if config.max_samples:
        ds = ds.select(range(min(len(ds), config.max_samples)))
    texts = []
    for item in ds:
        value = item.get(config.text_field)
        if value:
            texts.append(str(value))
    return texts


def load_texts(config: DataConfig, project_root: Path) -> list[str]:
    if config.dataset_name:
        return _read_hf_dataset(config)
    if not config.local_text_path:
        raise ValueError("Missing local_text_path")
    path = Path(config.local_text_path)
    if not path.is_absolute():
        path = project_root / path
    texts = _read_local_text(path)
    if config.max_samples:
        texts = texts[: config.max_samples]
    return texts


def cache_key(config: DataConfig, tokenizer_name: str, seq_len: int) -> str:
    payload = asdict(config)
    payload.update(
        {
            "tokenizer_name": tokenizer_name,
            "seq_len": seq_len,
            "cache_version": CACHE_VERSION,
        }
    )
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_token_blocks(texts: Iterable[str], tokenizer, seq_len: int) -> list[list[int]]:
    all_ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(int(x) for x in ids)
        if eos is not None:
            all_ids.append(int(eos))

    blocks = []
    for start in range(0, max(0, len(all_ids) - seq_len + 1), seq_len):
        block = all_ids[start : start + seq_len]
        if len(block) == seq_len:
            blocks.append(block)
    if not blocks and all_ids:
        pad = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
        block = all_ids[:seq_len] + [int(pad)] * max(0, seq_len - len(all_ids))
        blocks.append(block[:seq_len])
    return blocks


def load_or_build_blocks(config: DataConfig, tokenizer, seq_len: int, project_root: Path) -> tuple[list[list[int]], dict]:
    tokenizer_name = getattr(tokenizer, "name_or_path", "unknown-tokenizer")
    key = cache_key(config, tokenizer_name, seq_len)
    cache_dir = Path(config.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"blocks-{key}.json"

    if config.use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload["blocks"], {"cache": "hit", "cache_key": key, "cache_path": str(cache_path)}

    texts = load_texts(config, project_root)
    blocks = make_token_blocks(texts, tokenizer, seq_len)
    if not blocks:
        raise ValueError("No token blocks were produced")
    payload = {"blocks": blocks, "meta": {"cache_key": key, "num_blocks": len(blocks)}}
    if config.use_cache:
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return blocks, {"cache": "miss", "cache_key": key, "cache_path": str(cache_path), "num_texts": len(texts)}


def split_blocks(blocks: list[list[int]], validation_split: float, seed: int):
    rng = random.Random(seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    val_count = max(1, int(len(order) * validation_split)) if len(order) > 1 else 1
    val_ids = set(order[:val_count])
    train = [block for i, block in enumerate(blocks) if i not in val_ids]
    val = [block for i, block in enumerate(blocks) if i in val_ids]
    if not train:
        train = val
    return train, val


def build_dataloaders(config: DataConfig, tokenizer, seq_len: int, batch_size: int, project_root: Path):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required on the remote training server") from exc

    blocks, meta = load_or_build_blocks(config, tokenizer, seq_len, project_root)
    train_blocks, val_blocks = split_blocks(blocks, config.validation_split, config.seed)
    train_ds = TokenBlockDataset(train_blocks)
    val_ds = TokenBlockDataset(val_blocks)
    persistent = bool(config.persistent_workers and config.num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
    )
    meta.update(
        {
            "num_blocks": len(blocks),
            "train_blocks": len(train_ds),
            "val_blocks": len(val_ds),
        }
    )
    return train_loader, val_loader, meta
