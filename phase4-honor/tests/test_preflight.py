from pathlib import Path

from training_framework.config_merge import load_composed_config
from training_framework.preflight import (
    build_preflight_report,
    estimate_training_memory,
    oom_recommendations,
    render_preflight_markdown,
)


class TinyTokenizer:
    name_or_path = "unit-test-tokenizer"
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 128


def test_preflight_report_records_profiles_and_runtime():
    root = Path(__file__).resolve().parents[1]
    cfg = load_composed_config(
        "configs/base/causal_lm_debug.yaml",
        model_profile="configs/model_profiles/tiny_gpt2.yaml",
        data_profile="configs/data_profiles/local_fixture.yaml",
        overrides=["trainer.batch_size=3", "trainer.grad_accum_steps=2"],
        project_root=root,
    )
    report = build_preflight_report(
        cfg,
        root,
        tokenizer=TinyTokenizer(),
        parameter_count=1234,
        device="cpu",
    )
    assert report["metadata"]["model_profile"] == "tiny_gpt2"
    assert report["metadata"]["data_profile"] == "local_fixture"
    assert report["model"]["parameter_count"] == 1234
    assert report["trainer"]["tokens_per_step"] == cfg.trainer.seq_len * 3 * 2
    assert report["data"]["local_text_path_exists"] is True
    assert report["data"]["available_columns"] == ["text"]
    assert report["data"]["text_field_found"] is True
    assert report["memory_estimate"]["available"] is True
    assert report["memory_estimate"]["tokens_per_step"] == cfg.trainer.seq_len * 3 * 2


def test_preflight_markdown_is_human_readable():
    report = {
        "metadata": {"base_config": "base.yaml", "model_profile": "m", "data_profile": "d"},
        "model": {"name_or_path": "model", "parameter_count": 10},
        "data": {"source": "local_text", "dataset_name": None, "local_text_path": "fixtures/tiny_corpus.txt", "use_cache": True},
        "trainer": {"device": "cpu", "mixed_precision": "off", "tokens_per_step": 64},
        "memory_estimate": {
            "available": True,
            "parameter_mb": 1.0,
            "gradient_mb": 1.0,
            "optimizer_state_mb": 1.0,
            "activation_mb": 1.0,
            "predicted_allocated_peak_mb": 4.0,
            "predicted_reserved_peak_mb": 5.0,
            "memory_ratio_reserved": None,
        },
        "environment": {"cuda": {"available": False}},
        "warnings": [],
        "recommendations": [],
    }
    text = render_preflight_markdown(report)
    assert "# Preflight Report" in text
    assert "Tokens per optimizer step" in text
    assert "Available columns" in text
    assert "Memory Estimate" in text
    assert "Predicted reserved peak" in text
    assert "Recommendations" in text
    assert "- none" in text


def test_oom_recommendations_explain_batch_and_grad_accum_tradeoff():
    root = Path(__file__).resolve().parents[1]
    cfg = load_composed_config(
        "configs/base/causal_lm_debug.yaml",
        model_profile="configs/model_profiles/tiny_gpt2.yaml",
        data_profile="configs/data_profiles/local_fixture.yaml",
        overrides=["trainer.batch_size=8", "trainer.grad_accum_steps=2"],
        project_root=root,
    )
    recs = oom_recommendations(
        cfg,
        parameter_count=100_000_000,
        cuda={"available": True, "memory_total_gb": 1.0},
    )
    assert any("batch_size" in item for item in recs)
    assert any("Gradient accumulation" in item for item in recs)
    assert any("preflight-only" in item for item in recs)


def test_memory_estimate_explains_adafactor_vs_adamw():
    root = Path(__file__).resolve().parents[1]
    cfg = load_composed_config(
        "configs/base/causal_lm_debug.yaml",
        model_profile="configs/model_profiles/deepseek_placeholder.yaml",
        data_profile="configs/data_profiles/local_fixture.yaml",
        overrides=[
            "trainer.seq_len=2048",
            "trainer.batch_size=1",
            "trainer.grad_accum_steps=1",
            "trainer.mixed_precision=auto",
            "optimizer.name=adafactor",
        ],
        project_root=root,
    )
    cuda = {"available": True, "memory_total_gb": 23.55877685546875}
    adafactor = estimate_training_memory(cfg, parameter_count=1_346_471_936, cuda=cuda)
    cfg.optimizer.name = "adamw"
    adamw = estimate_training_memory(cfg, parameter_count=1_346_471_936, cuda=cuda)
    assert adafactor["predicted_reserved_peak_mb"] < adamw["predicted_reserved_peak_mb"]
    assert adafactor["memory_ratio_reserved"] < 1.0
    assert adamw["memory_ratio_reserved"] > 1.0
