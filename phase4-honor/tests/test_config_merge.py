from pathlib import Path

from training_framework.config_merge import load_composed_config, resolve_config_dict


def test_composed_profile_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_composed_config(
        "configs/base/causal_lm_debug.yaml",
        model_profile="configs/model_profiles/tiny_gpt2.yaml",
        data_profile="configs/data_profiles/local_fixture.yaml",
        overrides=["trainer.max_steps=2", "trainer.run_name=profile-smoke"],
        project_root=root,
    )
    assert cfg.metadata.model_profile == "tiny_gpt2"
    assert cfg.metadata.data_profile == "local_fixture"
    assert cfg.model.name_or_path == "sshleifer/tiny-gpt2"
    assert cfg.data.local_text_path == "fixtures/tiny_corpus.txt"
    assert cfg.trainer.max_steps == 2
    assert cfg.trainer.run_name == "profile-smoke"


def test_resolved_config_preserves_base_and_profile_notes():
    root = Path(__file__).resolve().parents[1]
    raw = resolve_config_dict(
        "configs/base/causal_lm_tinystories.yaml",
        model_profile="configs/model_profiles/distilgpt2.yaml",
        data_profile="configs/data_profiles/wikitext2.yaml",
        project_root=root,
    )
    assert raw["metadata"]["base_config"] == "configs/base/causal_lm_tinystories.yaml"
    assert raw["metadata"]["model_profile"] == "distilgpt2"
    assert raw["metadata"]["data_profile"] == "wikitext2"
    assert "model_profile:" in raw["metadata"]["notes"]
    assert raw["data"]["dataset_name"] == "wikitext"
    assert raw["data"]["dataset_config"] == "wikitext-2-raw-v1"
