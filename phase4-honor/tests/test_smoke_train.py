from pathlib import Path


def test_smoke_config_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs/debug.yaml").exists()
    assert (root / "train.py").exists()
