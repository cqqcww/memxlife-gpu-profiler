"""Configuration dataclasses and YAML loading.

The local laptop may not have PyYAML installed, so this module includes a tiny
fallback parser that supports the simple nested YAML/list shapes used by our
configs. On the course server, PyYAML will be used automatically when available.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text in {"", "null", "None", "~"}:
        return None
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _simple_yaml_load(path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by this project.

    This is intentionally not a full YAML parser. It supports nested mappings,
    scalar values, scalar lists, and lists of mappings such as the matrix files.
    The fallback keeps local scaffold tests useful even before PyYAML is
    installed on the laptop.
    """

    parsed_lines: list[tuple[int, str, int]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parsed_lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip(), line_no))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(parsed_lines):
            return {}, index

        current_indent, current_text, _ = parsed_lines[index]
        if current_indent < indent:
            return {}, index
        if current_indent != indent:
            _, _, line_no = parsed_lines[index]
            raise ValueError(f"{path}:{line_no}: unexpected indentation")

        if current_text.startswith("- "):
            values: list[Any] = []
            while index < len(parsed_lines):
                line_indent, text, line_no = parsed_lines[index]
                if line_indent != indent or not text.startswith("- "):
                    break
                item_text = text[2:].strip()
                index += 1

                if not item_text:
                    if index < len(parsed_lines) and parsed_lines[index][0] > indent:
                        child, index = parse_block(index, parsed_lines[index][0])
                        values.append(child)
                    else:
                        values.append(None)
                    continue

                if ":" in item_text:
                    key, value = item_text.split(":", 1)
                    item: dict[str, Any] = {}
                    if value.strip():
                        item[key] = _parse_scalar(value)
                    elif index < len(parsed_lines) and parsed_lines[index][0] > indent:
                        child, index = parse_block(index, parsed_lines[index][0])
                        item[key] = child
                    else:
                        item[key] = None

                    if index < len(parsed_lines) and parsed_lines[index][0] > indent:
                        child, index = parse_block(index, parsed_lines[index][0])
                        if not isinstance(child, dict):
                            raise ValueError(
                                f"{path}:{line_no}: list mapping child must be a mapping"
                            )
                        item.update(child)
                    values.append(item)
                else:
                    values.append(_parse_scalar(item_text))
            return values, index

        mapping: dict[str, Any] = {}
        while index < len(parsed_lines):
            line_indent, text, line_no = parsed_lines[index]
            if line_indent != indent or text.startswith("- "):
                break
            if ":" not in text:
                raise ValueError(f"{path}:{line_no}: expected key: value")
            key, value = text.split(":", 1)
            index += 1
            if value.strip():
                mapping[key] = _parse_scalar(value)
            elif index < len(parsed_lines) and parsed_lines[index][0] > indent:
                child, index = parse_block(index, parsed_lines[index][0])
                mapping[key] = child
            else:
                mapping[key] = None
        return mapping, index

    if not parsed_lines:
        return {}
    data, final_index = parse_block(0, parsed_lines[0][0])
    if final_index != len(parsed_lines):
        _, _, line_no = parsed_lines[final_index]
        raise ValueError(f"{path}:{line_no}: could not parse remaining YAML")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    return data


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    try:
        import yaml  # type: ignore

        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{cfg_path} must contain a mapping at top level")
        return data
    except ModuleNotFoundError:
        text = cfg_path.read_text(encoding="utf-8")
        if text.lstrip().startswith("{"):
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"{cfg_path} must contain a mapping at top level")
            return data
        return _simple_yaml_load(cfg_path)


@dataclass
class MetadataConfig:
    name: str = ""
    tags: str = ""
    notes: str = ""
    base_config: str = ""
    model_profile: str = ""
    data_profile: str = ""


@dataclass
class ModelConfig:
    name_or_path: str = "sshleifer/tiny-gpt2"
    tokenizer_name: str | None = None
    from_pretrained: bool = False
    trust_remote_code: bool = False
    dtype: str = "auto"
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_field: str = "text"
    local_text_path: str | None = "fixtures/tiny_corpus.txt"
    cache_dir: str = "data_cache"
    use_cache: bool = True
    validation_split: float = 0.1
    max_samples: int = 128
    seed: int = 1337
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "warmup_cosine"
    warmup_steps: int = 5
    min_lr_ratio: float = 0.1


@dataclass
class TrainerConfig:
    run_name: str = "debug"
    output_dir: str = "runs"
    seq_len: int = 64
    batch_size: int = 2
    max_steps: int = 8
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    log_every_steps: int = 1
    validate_every_steps: int = 4
    mixed_precision: str = "auto"
    device: str = "auto"
    compile_model: bool = False
    resume_from: str | None = None


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    jsonl: bool = True
    console: bool = True


@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_every_steps: int = 4
    keep_last: int = 2
    save_rng: bool = True


@dataclass
class AgentConfig:
    enabled: bool = True
    proposal_mode: bool = True
    ledger_path: str = "runs/ledger.jsonl"


@dataclass
class ExperimentConfig:
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def validate(self) -> None:
        if not self.model.name_or_path:
            raise ValueError("model.name_or_path is required")
        if not self.data.dataset_name and not self.data.local_text_path:
            raise ValueError("data.dataset_name or data.local_text_path is required")
        if self.trainer.max_steps <= 0:
            raise ValueError("trainer.max_steps must be > 0")
        if self.trainer.seq_len <= 0:
            raise ValueError("trainer.seq_len must be > 0")
        if self.trainer.batch_size <= 0:
            raise ValueError("trainer.batch_size must be > 0")
        if self.trainer.grad_accum_steps <= 0:
            raise ValueError("trainer.grad_accum_steps must be > 0")
        if self.optimizer.lr <= 0:
            raise ValueError("optimizer.lr must be > 0")
        if self.checkpoint.enabled and self.checkpoint.save_every_steps <= 0:
            raise ValueError("checkpoint.save_every_steps must be > 0")
        if not 0 < self.data.validation_split < 0.9:
            raise ValueError("data.validation_split must be between 0 and 0.9")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_tuple(value: Any) -> tuple[float, float]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if isinstance(value, str) and "," in value:
        left, right = value.split(",", 1)
        return (float(left.strip()), float(right.strip()))
    raise ValueError("optimizer.betas must be two floats")


def _section(cls: type, raw: dict[str, Any] | None):
    raw = dict(raw or {})
    if cls is OptimizerConfig and "betas" in raw:
        raw["betas"] = _coerce_tuple(raw["betas"])
    if cls is TrainerConfig and "mixed_precision" in raw:
        # PyYAML's YAML 1.1 rules parse unquoted "off" as False. Normalize it
        # back to the string enum the trainer expects.
        if raw["mixed_precision"] is False:
            raw["mixed_precision"] = "off"
        elif raw["mixed_precision"] is True:
            raw["mixed_precision"] = "auto"
    allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig(
        metadata=_section(MetadataConfig, raw.get("metadata")),
        model=_section(ModelConfig, raw.get("model")),
        data=_section(DataConfig, raw.get("data")),
        optimizer=_section(OptimizerConfig, raw.get("optimizer")),
        scheduler=_section(SchedulerConfig, raw.get("scheduler")),
        trainer=_section(TrainerConfig, raw.get("trainer")),
        logging=_section(LoggingConfig, raw.get("logging")),
        checkpoint=_section(CheckpointConfig, raw.get("checkpoint")),
        agent=_section(AgentConfig, raw.get("agent")),
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml_dict(path)
    return config_from_dict(raw)


def dump_yaml_dict(data: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except ModuleNotFoundError:
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    except ModuleNotFoundError:
        out.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
