import numpy as np
import pytest

from hw9.text_process import TextProcess

CORPUS = [
    "The cat sat on the mat.",
    "The dog chased the cat!",
    "Dogs and cats don't always get along.",
    "Is the dog happy? Yes, the dog is very happy.",
    "Machine learning turns text into numbers; numbers feed neural networks.",
]


@pytest.fixture
def tp():
    return TextProcess(mode="word").fit(CORPUS)


def test_reserved_ids(tp):
    assert tp.word_index["<PAD>"] == 0 and tp.word_index["<OOV>"] == 1


def test_most_frequent_token_gets_smallest_id(tp):
    assert tp.word_index["the"] == 2


def test_index_word_is_inverse(tp):
    assert all(tp.index_word[i] == w for w, i in tp.word_index.items())


def test_encode_returns_ints_one_per_token(tp):
    e = tp.encode("The cat sat.")
    assert len(e) == 4 and all(isinstance(i, int) for i in e)


def test_encode_is_case_insensitive(tp):
    assert tp.encode("THE CAT") == tp.encode("the cat")


def test_unknown_word_maps_to_oov(tp):
    assert tp.encode("zebra") == [1]
    assert tp.decode([1]) == "<OOV>"


@pytest.mark.parametrize("sentence", CORPUS)
def test_round_trip_word_mode(tp, sentence):
    assert tp.decode(tp.encode(sentence)) == tp.normalize(sentence)


def test_batch_padding_and_decode(tp):
    X = tp.encode_batch(["the cat", "the dog chased the cat"])
    assert X.shape == (2, 5) and X[0, 2:].tolist() == [0, 0, 0]
    assert tp.decode_batch(X) == ["the cat", "the dog chased the cat"]


def test_batch_truncation(tp):
    assert tp.encode_batch(["the dog chased the cat"], maxlen=3).shape == (1, 3)


def test_one_hot_round_trip(tp):
    oh = tp.one_hot_encode("the dog is happy.")
    assert oh.shape == (5, tp.vocab_size) and (oh.sum(1) == 1).all()
    assert tp.one_hot_decode(oh) == "the dog is happy."


def test_char_mode_is_lossless():
    c = TextProcess(mode="char", lower=False).fit(["Hello, World! 123"])
    assert c.decode(c.encode("Hello, World! 123")) == "Hello, World! 123"


def test_json_round_trip(tp):
    other = TextProcess.from_json(tp.to_json())
    assert other.encode("the dog") == tp.encode("the dog")


def test_invalid_mode():
    with pytest.raises(AssertionError):
        TextProcess(mode="bogus")
