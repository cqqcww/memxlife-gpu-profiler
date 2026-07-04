# Preflight Report

## Profiles

- Base config: `configs/base/causal_lm_tinystories.yaml`
- Model profile: `qwen_small_placeholder`
- Data profile: `tinystories`
- Notes: Base for small HuggingFace text dataset experiments. || model_profile: family=qwen-causal-lm | expected_memory=unknown | Stretch profile. Requires download, tokenizer compatibility, and memory preflight. | Try only after GPT-style profile switching is validated. || data_profile: Small TinyStories subset for fast, coherent causal-LM experiments.

## Model And Data

- Model: `Qwen/Qwen2.5-0.5B`
- Parameters: `494032768`
- Data source: `huggingface_dataset`
- Dataset/local path: `roneneldan/TinyStories`
- Available columns: `['text']`
- Text field found: `True`
- Cache enabled: `True`

## Runtime

- Device: `cuda`
- Mixed precision: `auto`
- Tokens per optimizer step: `64`
- CUDA available: `True`

## Recommendations

- none

## Warnings

- trust_remote_code is enabled; this is acceptable for stretch profiles but should be explicit
