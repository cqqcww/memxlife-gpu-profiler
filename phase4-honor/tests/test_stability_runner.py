import json
from pathlib import Path

from agent.stability_runner import (
    load_recommendation,
    materialize_stability_config,
    selected_record,
    summarize_stability,
)
from training_framework.config import dump_yaml_dict, load_config


def _write_base_config(path: Path) -> None:
    dump_yaml_dict(
        {
            "metadata": {"name": "unit"},
            "model": {"name_or_path": "sshleifer/tiny-gpt2"},
            "data": {"local_text_path": "fixtures/tiny_corpus.txt", "validation_split": 0.2},
            "optimizer": {"name": "adafactor", "lr": 0.0001},
            "trainer": {
                "run_name": "probe",
                "seq_len": 2048,
                "batch_size": 1,
                "max_steps": 3,
                "validate_every_steps": 3,
            },
            "checkpoint": {"enabled": False},
        },
        path,
    )


def test_stability_runner_materializes_config_from_recommendation(tmp_path):
    config_path = tmp_path / "selected.json"
    _write_base_config(config_path)
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_text(
        json.dumps(
            {
                "probe": "unit_probe",
                "records": [
                    {
                        "variant": "tok2048",
                        "token_budget": 2048,
                        "config": str(config_path),
                        "status": 0,
                    }
                ],
                "recommendation": {
                    "selected_variant": "tok2048",
                    "selected_token_budget": 2048,
                    "action": "stability_run",
                },
            }
        ),
        encoding="utf-8",
    )
    payload = load_recommendation(recommendation_path, project_root=tmp_path)
    assert selected_record(payload)["variant"] == "tok2048"
    stability_config = materialize_stability_config(payload, steps=50, project_root=tmp_path)
    cfg = load_config(stability_config)
    assert cfg.trainer.seq_len == 2048
    assert cfg.trainer.max_steps == 50
    assert cfg.trainer.validate_every_steps == 10
    assert cfg.trainer.run_name == "probe-stability-50step"
    assert cfg.checkpoint.enabled is False


def test_summarize_stability_passes_complete_finite_run(tmp_path):
    run_dir = tmp_path / "runs" / "stable"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "train", "metrics": {"loss": 5.0, "tokens_per_sec": 10.0}}),
                json.dumps({"event": "train", "metrics": {"loss": 4.0, "tokens_per_sec": 12.0}}),
                json.dumps({"event": "validate", "metrics": {"loss": 3.0}}),
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"global_step": 2}), encoding="utf-8")
    stability = summarize_stability(run_dir, 0, {"requested_steps": 2})
    assert stability["status"] == "pass"
    assert stability["completed_train_steps"] == 2
    assert stability["logged_train_events"] == 2
    assert stability["last_val_loss"] == 3.0


def test_summarize_stability_uses_global_step_over_logged_event_count(tmp_path):
    run_dir = tmp_path / "runs" / "sparse_logs"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "train", "metrics": {"loss": 5.0, "tokens_per_sec": 10.0}}),
                json.dumps({"event": "validate", "metrics": {"loss": 3.0}}),
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"global_step": 50}), encoding="utf-8")
    stability = summarize_stability(run_dir, 0, {"requested_steps": 50})
    assert stability["status"] == "pass"
    assert stability["completed_train_steps"] == 50
    assert stability["logged_train_events"] == 1


def test_summarize_stability_fails_nonfinite_loss(tmp_path):
    run_dir = tmp_path / "runs" / "unstable"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        json.dumps({"event": "train", "metrics": {"loss": float("nan"), "tokens_per_sec": 10.0}}),
        encoding="utf-8",
    )
    stability = summarize_stability(run_dir, 0, {"requested_steps": 1})
    assert stability["status"] == "fail"
    assert any("non-finite" in reason for reason in stability["reasons"])
