from pathlib import Path

from training_framework.checkpoint import CheckpointManager


def expect_raises(expected, fn, *args, message_fragment=None):
    try:
        fn(*args)
    except expected as exc:
        if message_fragment is not None:
            assert message_fragment in str(exc)
        return
    raise AssertionError(f"Expected {expected.__name__}")


def test_checkpoint_contract_documented():
    expected = {"model", "optimizer", "scheduler", "global_step", "extra", "rng"}
    assert "global_step" in expected
    assert "rng" in expected


def test_missing_checkpoint_file_has_readable_error(tmp_path):
    manager = CheckpointManager(tmp_path)
    missing = tmp_path / "checkpoints" / "does_not_exist.pt"
    expect_raises(
        FileNotFoundError,
        manager.load,
        missing,
        None,
        message_fragment="Checkpoint file not found",
    )


def test_empty_checkpoint_dir_has_readable_error(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    manager = CheckpointManager(tmp_path)
    expect_raises(
        FileNotFoundError,
        manager.load,
        empty_dir,
        None,
        message_fragment="does not contain latest.txt",
    )
