import json
from pathlib import Path

from agent.recommendation_report import (
    build_recommendation_section,
    render_markdown,
    write_recommendation_report,
)


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_recommendation_report_calibrates_memory_and_renders(tmp_path):
    run_dir = tmp_path / "runs" / "deepseek-realdata"
    _write_json(
        run_dir / "summary.json",
        {
            "cuda_memory": {
                "cuda_peak_allocated_mb": 14305.0,
                "cuda_peak_reserved_mb": 18828.0,
            }
        },
    )
    _write_json(
        run_dir / "preflight.json",
        {
            "metadata": {"data_profile": "wikitext2"},
            "model": {
                "name_or_path": "deepseek-ai/deepseek-coder-1.3b-base",
                "gradient_checkpointing": False,
            },
            "trainer": {
                "tokens_per_step": 2048,
                "seq_len": 2048,
                "batch_size": 1,
                "grad_accum_steps": 1,
                "mixed_precision": "auto",
            },
            "optimizer": {"name": "adafactor"},
            "memory_estimate": {
                "predicted_allocated_peak_mb": 15000.0,
                "predicted_reserved_peak_mb": 18750.0,
                "memory_ratio_reserved": 0.78,
            },
        },
    )
    probe = {
        "recommendation": {
            "action": "stability_run",
            "reason": "run a longer stability check",
        }
    }
    stability = {
        "stability": {
            "run_dir": str(run_dir),
            "status": "pass",
            "completed_train_steps": 50,
            "requested_steps": 50,
            "avg_tokens_per_sec": 3655.35,
            "first_train_loss": 8.9343,
            "last_train_loss": 6.4882,
            "last_val_loss": 6.9378,
            "cuda_peak_allocated_mb": 14305.0,
            "cuda_peak_reserved_mb": 18828.0,
        }
    }
    section = build_recommendation_section(
        probe_payload=probe,
        stability_payload=stability,
        project_root=tmp_path,
    )
    assert section["decision"]["promote_for_longer_probe"] is True
    assert abs(section["memory_calibration"]["reserved_error_pct"]) < 1.0
    text = render_markdown(section)
    assert "Current Phase 4 Recommendation" in text
    assert "Memory Calibration" in text
    assert "Promote for longer probe" in text


def test_write_recommendation_report_outputs_json_and_markdown(tmp_path):
    run_dir = tmp_path / "runs" / "deepseek-realdata"
    _write_json(run_dir / "summary.json", {"cuda_memory": {}})
    _write_json(
        run_dir / "preflight.json",
        {
            "metadata": {"data_profile": "wikitext2"},
            "model": {"name_or_path": "m", "gradient_checkpointing": False},
            "trainer": {"tokens_per_step": 1, "seq_len": 1, "batch_size": 1, "grad_accum_steps": 1},
            "optimizer": {"name": "adafactor"},
            "memory_estimate": {},
        },
    )
    probe_path = tmp_path / "probe.json"
    stability_path = tmp_path / "stability.json"
    _write_json(probe_path, {"recommendation": {}})
    _write_json(
        stability_path,
        {
            "stability": {
                "run_dir": str(run_dir),
                "status": "pass",
                "completed_train_steps": 1,
                "requested_steps": 1,
            }
        },
    )
    artifacts = write_recommendation_report(
        probe_path=probe_path,
        stability_path=stability_path,
        output_base=tmp_path / "out" / "recommendation",
        project_root=tmp_path,
    )
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["md"]).exists()
