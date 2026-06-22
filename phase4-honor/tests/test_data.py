from training_framework.data import cache_key, make_token_blocks, split_blocks
from training_framework.config import DataConfig


class ToyTokenizer:
    name_or_path = "toy"
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 31 + 1 for ch in text]


def test_cache_key_changes_with_seq_len():
    cfg = DataConfig(local_text_path="fixtures/tiny_corpus.txt")
    assert cache_key(cfg, "toy", 8) != cache_key(cfg, "toy", 16)


def test_cache_key_changes_with_tokenizer_and_samples():
    cfg = DataConfig(local_text_path="fixtures/tiny_corpus.txt", max_samples=8)
    more = DataConfig(local_text_path="fixtures/tiny_corpus.txt", max_samples=16)
    assert cache_key(cfg, "toy-a", 8) != cache_key(cfg, "toy-b", 8)
    assert cache_key(cfg, "toy-a", 8) != cache_key(more, "toy-a", 8)


def test_make_token_blocks_shape():
    blocks = make_token_blocks(["hello world"], ToyTokenizer(), seq_len=8)
    assert blocks
    assert all(len(block) == 8 for block in blocks)


def test_split_blocks_keeps_train_and_validation():
    blocks = [[i] * 4 for i in range(10)]
    train, val = split_blocks(blocks, validation_split=0.2, seed=7)
    assert train
    assert val
    assert len(train) + len(val) == len(blocks)
