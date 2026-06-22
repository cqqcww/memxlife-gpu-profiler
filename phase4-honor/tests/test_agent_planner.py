from pathlib import Path

from agent.planner import choose_next_config, completed_configs, infer_bottleneck, load_ledger
from agent.runner import latest_run


def test_choose_next_config_uses_ladder_order():
    assert choose_next_config(set()) == "configs/debug.yaml"
    assert (
        choose_next_config({"configs/debug.yaml"})
        == "configs/baseline_tinystories.yaml"
    )


def test_completed_configs_only_counts_successes():
    records = [
        {"config": "configs/debug.yaml", "status": 0},
        {"config": "configs/baseline_tinystories.yaml", "status": 1},
    ]
    assert completed_configs(records) == {"configs/debug.yaml"}


def test_load_ledger_missing_returns_empty(tmp_path):
    assert load_ledger(tmp_path / "missing.jsonl") == []


def test_latest_run_ignores_ledger_file(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "ledger.jsonl").write_text("{}", encoding="utf-8")
    (runs / "matrix_logs").mkdir()
    first = runs / "debug-1"
    first.mkdir()
    (first / "copied_config.yaml").write_text("{}", encoding="utf-8")
    assert latest_run(tmp_path) == first


def test_infer_bottleneck_detects_optimizer_overhead():
    bottleneck, proposal = infer_bottleneck(
        {
            "last_train": {
                "data_s": 0.0001,
                "forward_s": 0.01,
                "backward_s": 0.01,
                "optimizer_s": 0.009,
            }
        }
    )
    assert "optimizer" in bottleneck
    assert "batch" in proposal
