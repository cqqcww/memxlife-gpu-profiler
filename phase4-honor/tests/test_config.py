from pathlib import Path

from training_framework.config import load_config


def expect_raises(expected, fn, *args):
    try:
        fn(*args)
    except expected:
        return
    raise AssertionError(f"Expected {expected.__name__}")


def test_debug_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/debug.yaml")
    assert cfg.model.name_or_path
    assert cfg.trainer.max_steps > 0
    assert cfg.optimizer.betas == (0.9, 0.95)


def test_bad_config_unknown_key_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  name_or_path: sshleifer/tiny-gpt2",
                "trainer:",
                "  max_steps: 1",
                "  definitely_not_a_real_key: true",
            ]
        ),
        encoding="utf-8",
    )
    expect_raises(ValueError, load_config, path)


def test_bad_config_invalid_steps_fails(tmp_path):
    path = tmp_path / "bad_steps.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  name_or_path: sshleifer/tiny-gpt2",
                "data:",
                "  local_text_path: fixtures/tiny_corpus.txt",
                "trainer:",
                "  max_steps: 0",
            ]
        ),
        encoding="utf-8",
    )
    expect_raises(ValueError, load_config, path)


def test_yaml_boolean_off_normalizes_to_mixed_precision_string(tmp_path):
    path = tmp_path / "off.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  name_or_path: sshleifer/tiny-gpt2",
                "data:",
                "  local_text_path: fixtures/tiny_corpus.txt",
                "trainer:",
                "  mixed_precision: false",
                "  max_steps: 1",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.trainer.mixed_precision == "off"


def test_checkpoint_interval_can_be_zero_when_disabled(tmp_path):
    path = tmp_path / "no_checkpoint.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  name_or_path: sshleifer/tiny-gpt2",
                "data:",
                "  local_text_path: fixtures/tiny_corpus.txt",
                "trainer:",
                "  max_steps: 1",
                "checkpoint:",
                "  enabled: false",
                "  save_every_steps: 0",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.checkpoint.enabled is False
