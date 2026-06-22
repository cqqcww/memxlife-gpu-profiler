"""Model and tokenizer construction."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig


@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    parameter_count: int


def build_model_and_tokenizer(config: ModelConfig) -> ModelBundle:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers is required on the remote training server"
        ) from exc

    tokenizer_name = config.tokenizer_name or config.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.trust_remote_code,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
    else:
        hf_config = AutoConfig.from_pretrained(
            config.name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if getattr(hf_config, "pad_token_id", None) is None:
            hf_config.pad_token_id = tokenizer.pad_token_id
        model = AutoModelForCausalLM.from_config(hf_config)

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    parameter_count = sum(p.numel() for p in model.parameters())
    return ModelBundle(model=model, tokenizer=tokenizer, parameter_count=parameter_count)
