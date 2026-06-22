from pathlib import Path

from agent.auto_probe import (
    build_recommendation,
    load_probe,
    materialize_probe_config,
    render_recommendation_markdown,
    token_budget_overrides,
)
from training_framework.config import load_config


def test_auto_probe_materializes_token_budget_config():
    root = Path(__file__).resolve().parents[1]
    probe = load_probe("configs/auto_probes/deepseek_adafactor_token_budget.yaml", project_root=root)
    overrides = token_budget_overrides(probe, 256)
    assert overrides["trainer.seq_len"] == 256
    assert overrides["trainer.batch_size"] == 1
    assert overrides["optimizer.name"] == "adafactor"
    assert overrides["checkpoint.enabled"] is False

    cfg = load_config(materialize_probe_config(probe, 256, project_root=root))
    assert cfg.metadata.model_profile == "deepseek_placeholder"
    assert cfg.trainer.seq_len == 256
    assert cfg.trainer.batch_size == 1
    assert cfg.optimizer.name == "adafactor"
    assert cfg.checkpoint.enabled is False
    assert cfg.data.local_text_path == "fixtures/medium_corpus.txt"


def test_realdata_auto_probe_uses_wikitext_profile():
    root = Path(__file__).resolve().parents[1]
    probe = load_probe("configs/auto_probes/deepseek_adafactor_wikitext_realdata.yaml", project_root=root)
    cfg = load_config(materialize_probe_config(probe, 2048, project_root=root))
    assert cfg.metadata.model_profile == "deepseek_placeholder"
    assert cfg.metadata.data_profile == "wikitext2"
    assert cfg.data.dataset_name == "wikitext"
    assert cfg.data.local_text_path is None
    assert cfg.trainer.seq_len == 2048
    assert cfg.optimizer.name == "adafactor"
    assert cfg.checkpoint.enabled is False


def test_auto_probe_recommends_larger_budget_when_memory_headroom_remains():
    probe = {
        "name": "unit",
        "memory_limit_mb": 24576,
        "headroom_ratio_for_expand": 0.75,
        "min_speedup_ratio": 1.03,
    }
    records = [
        {
            "variant": "tok128",
            "token_budget": 128,
            "status": 0,
            "avg_tokens_per_sec": 380.0,
            "last_val_loss": 10.0,
            "cuda_peak_reserved_mb": 12500.0,
        },
        {
            "variant": "tok256",
            "token_budget": 256,
            "status": 0,
            "avg_tokens_per_sec": 760.0,
            "last_val_loss": 9.5,
            "cuda_peak_reserved_mb": 13000.0,
        },
    ]
    recommendation = build_recommendation(probe, records)
    assert recommendation["action"] == "try_larger_budget"
    assert recommendation["next_token_budget"] == 512
    assert recommendation["selected_token_budget"] == 256


def test_auto_probe_promotes_last_safe_after_oom():
    probe = {"name": "unit", "memory_limit_mb": 24576}
    records = [
        {
            "variant": "tok256",
            "token_budget": 256,
            "status": 0,
            "avg_tokens_per_sec": 760.0,
            "last_val_loss": 9.5,
            "cuda_peak_reserved_mb": 13000.0,
        },
        {
            "variant": "tok512",
            "token_budget": 512,
            "status": 1,
            "failure_kind": "cuda_oom",
        },
    ]
    recommendation = build_recommendation(probe, records)
    assert recommendation["action"] == "promote_last_safe"
    assert recommendation["selected_token_budget"] == 256
    assert recommendation["next_token_budget"] is None
    text = render_recommendation_markdown(probe, records, recommendation)
    assert "promote_last_safe" in text
    assert "cuda_oom" in text
