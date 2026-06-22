from pathlib import Path
import math

from agent.matrix_runner import (
    choose_best_record,
    classify_failure,
    expand_variants,
    load_matrix,
    materialize_variant_config,
    render_matrix_markdown,
    variant_complexity,
)
from training_framework.config import load_config


def test_matrix_expansion_and_materialization():
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix("configs/matrices/cache_on_off.yaml", project_root=root)
    variants = expand_variants(matrix)
    assert [variant["name"] for variant in variants] == ["cache_off", "cache_on"]
    config_path = materialize_variant_config(matrix, variants[0], project_root=root)
    cfg = load_config(config_path)
    assert cfg.trainer.run_name == "sweep-cache-off"
    assert cfg.data.use_cache is False
    assert cfg.trainer.max_steps == 60


def test_profile_matrix_materializes_model_and_data_profiles():
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix("configs/matrices/qwen_throughput_probe.yaml", project_root=root)
    variants = expand_variants(matrix)
    config_path = materialize_variant_config(matrix, variants[1], project_root=root)
    cfg = load_config(config_path)
    assert cfg.metadata.model_profile == "qwen_small_placeholder"
    assert cfg.metadata.data_profile == "tinystories"
    assert cfg.model.name_or_path == "Qwen/Qwen2.5-0.5B"
    assert cfg.model.gradient_checkpointing is False
    assert cfg.trainer.seq_len == 64
    assert cfg.trainer.batch_size == 1
    assert cfg.data.dataset_name == "roneneldan/TinyStories"


def test_matrix_variant_can_request_preflight_only():
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix("configs/matrices/deepseek_safety_probe.yaml", project_root=root)
    variants = expand_variants(matrix)
    assert variants[0]["preflight_only"] is True
    assert variants[1]["preflight_only"] is False
    config_path = materialize_variant_config(matrix, variants[0], project_root=root)
    cfg = load_config(config_path)
    assert cfg.metadata.model_profile == "deepseek_placeholder"
    assert cfg.trainer.seq_len == 16
    adafactor_cfg = load_config(materialize_variant_config(matrix, variants[2], project_root=root))
    assert adafactor_cfg.optimizer.name == "adafactor"
    assert adafactor_cfg.checkpoint.enabled is False
    assert adafactor_cfg.trainer.validate_every_steps == 99


def test_deepseek_adafactor_probe_materializes_low_memory_variants():
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix("configs/matrices/deepseek_adafactor_probe.yaml", project_root=root)
    variants = expand_variants(matrix)
    assert [variant["name"] for variant in variants] == ["s16_b1_5step_val", "s32_b1_3step_val"]
    cfg = load_config(materialize_variant_config(matrix, variants[0], project_root=root))
    assert cfg.metadata.model_profile == "deepseek_placeholder"
    assert cfg.optimizer.name == "adafactor"
    assert cfg.checkpoint.enabled is False
    assert cfg.trainer.max_steps == 5
    assert cfg.trainer.validate_every_steps == 5


def test_deepseek_adafactor_scale_probe_compares_checkpointing_modes():
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix("configs/matrices/deepseek_adafactor_scale_probe.yaml", project_root=root)
    variants = expand_variants(matrix)
    assert [variant["name"] for variant in variants] == [
        "s64_b1_gc_on_3step_val",
        "s64_b1_gc_off_3step_val",
    ]
    gc_on = load_config(materialize_variant_config(matrix, variants[0], project_root=root))
    gc_off = load_config(materialize_variant_config(matrix, variants[1], project_root=root))
    assert gc_on.trainer.seq_len == 64
    assert gc_on.model.gradient_checkpointing is True
    assert gc_off.model.gradient_checkpointing is False
    assert gc_off.optimizer.name == "adafactor"
    assert gc_off.checkpoint.enabled is False


def test_deepseek_adafactor_budget_probe_uses_medium_fixture_for_batch_compare():
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix("configs/matrices/deepseek_adafactor_budget_probe.yaml", project_root=root)
    variants = expand_variants(matrix)
    assert [variant["name"] for variant in variants] == [
        "s128_b1_gc_off_3step_val",
        "s64_b2_gc_off_3step_val",
    ]
    seq_cfg = load_config(materialize_variant_config(matrix, variants[0], project_root=root))
    batch_cfg = load_config(materialize_variant_config(matrix, variants[1], project_root=root))
    assert seq_cfg.trainer.seq_len == 128
    assert seq_cfg.trainer.batch_size == 1
    assert batch_cfg.trainer.seq_len == 64
    assert batch_cfg.trainer.batch_size == 2
    assert batch_cfg.data.local_text_path == "fixtures/medium_corpus.txt"
    assert batch_cfg.model.gradient_checkpointing is False


def test_deepseek_adafactor_256_probe_compares_three_token_shapes():
    root = Path(__file__).resolve().parents[1]
    matrix = load_matrix("configs/matrices/deepseek_adafactor_256_probe.yaml", project_root=root)
    variants = expand_variants(matrix)
    assert [variant["name"] for variant in variants] == [
        "s256_b1_gc_off_3step_val",
        "s128_b2_gc_off_3step_val",
        "s64_b4_gc_off_3step_val",
    ]
    cfgs = [load_config(materialize_variant_config(matrix, variant, project_root=root)) for variant in variants]
    assert [cfg.trainer.seq_len * cfg.trainer.batch_size for cfg in cfgs] == [256, 256, 256]
    assert all(cfg.optimizer.name == "adafactor" for cfg in cfgs)
    assert all(cfg.checkpoint.enabled is False for cfg in cfgs)
    assert all(cfg.data.local_text_path == "fixtures/medium_corpus.txt" for cfg in cfgs)


def test_matrix_best_config_reasoning_prefers_fast_validation_sane_candidate():
    records = [
        {
            "variant": "bs2_ga1",
            "status": 0,
            "avg_tokens_per_sec": 100.0,
            "last_val_loss": 5.0,
            "complexity": variant_complexity({"trainer.batch_size": 2, "trainer.grad_accum_steps": 1}),
        },
        {
            "variant": "bs8_ga1",
            "status": 0,
            "avg_tokens_per_sec": 150.0,
            "last_val_loss": 5.2,
            "complexity": variant_complexity({"trainer.batch_size": 8, "trainer.grad_accum_steps": 1}),
        },
        {
            "variant": "fast_bad_val",
            "status": 0,
            "avg_tokens_per_sec": 180.0,
            "last_val_loss": 8.0,
            "complexity": variant_complexity({"trainer.batch_size": 16, "trainer.grad_accum_steps": 1}),
        },
    ]
    selection = choose_best_record(records)
    assert selection["variant"] == "bs8_ga1"
    text = render_matrix_markdown({"name": "unit", "description": "test"}, records, selection)
    assert "Best Config Reasoning" in text
    assert "bs8_ga1" in text


def test_matrix_best_config_reasoning_ignores_nan_validation_loss():
    records = [
        {
            "variant": "unstable_fast",
            "status": 0,
            "avg_tokens_per_sec": 200.0,
            "last_val_loss": math.nan,
            "complexity": variant_complexity({"trainer.batch_size": 1, "trainer.grad_accum_steps": 1}),
        },
        {
            "variant": "stable",
            "status": 0,
            "avg_tokens_per_sec": 150.0,
            "last_val_loss": 8.9,
            "complexity": variant_complexity({"trainer.batch_size": 2, "trainer.grad_accum_steps": 1}),
        },
    ]
    selection = choose_best_record(records)
    assert selection["variant"] == "stable"
    assert "nan" not in selection["reason"].lower()


def test_classify_failure_detects_cuda_oom():
    failure = classify_failure(1, "torch.cuda.OutOfMemoryError: CUDA out of memory")
    assert failure["failure_kind"] == "cuda_oom"
    assert "optimizer" in failure["failure_reason"]


def test_matrix_markdown_marks_preflight_execution():
    text = render_matrix_markdown(
        {"name": "unit", "description": "test"},
        [
            {
                "variant": "preflight",
                "execution": "preflight_only",
                "status": 0,
                "complexity": {"label": "x"},
            }
        ],
    )
    assert "Execution" in text
    assert "preflight_only" in text


def test_matrix_markdown_includes_peak_cuda_memory():
    text = render_matrix_markdown(
        {"name": "unit", "description": "test"},
        [
            {
                "variant": "mem",
                "execution": "train",
                "status": 0,
                "avg_tokens_per_sec": 1.0,
                "last_val_loss": 2.0,
                "cuda_peak_allocated_mb": 123.4,
                "cuda_peak_reserved_mb": 256.7,
                "complexity": {"label": "x"},
            }
        ],
    )
    assert "Peak CUDA MB" in text
    assert "123/257" in text


def test_variant_complexity_includes_sequence_and_checkpointing():
    complexity = variant_complexity(
        {
            "trainer.batch_size": 1,
            "trainer.grad_accum_steps": 1,
            "trainer.seq_len": 64,
            "model.gradient_checkpointing": True,
        }
    )
    assert "seq_len=64" in complexity["label"]
    assert "gradient_checkpointing=True" in complexity["label"]
    assert complexity["penalty"] == 2
