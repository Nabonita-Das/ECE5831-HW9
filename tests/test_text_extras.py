import numpy as np

from hw9.text_extras import (BPE, TextVectorizer, hash_encode, make_toy_sentiment,
                             train_embedding_classifier)
from hw9.text_process import TextProcess

DOCS = ["the cat sat on the mat", "the dog chased the cat", "dogs and cats are friends"]


def test_count_vectorizer_counts_terms():
    v = TextVectorizer("count").fit(DOCS)
    X = v.transform(["the cat the dog"])
    names = v.feature_names()
    assert X[0, names.index("the")] == 2 and X[0, names.index("cat")] == 1 and X[0, names.index("mat")] == 0


def test_binary_vectorizer_is_zero_one():
    X = TextVectorizer("binary").fit_transform(DOCS)
    assert set(np.unique(X)) <= {0.0, 1.0}


def test_tfidf_rows_are_unit_length_and_rare_words_weigh_more():
    v = TextVectorizer("tfidf")
    X = v.fit_transform(DOCS)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    names = v.feature_names()
    assert X[0, names.index("mat")] > X[0, names.index("the")] * 0.5        # 'mat' is rarer than 'the'
    assert v.idf_[names.index("mat")] > v.idf_[names.index("the")]


def test_bigrams_add_features():
    uni = TextVectorizer("count", (1, 1)).fit(DOCS)
    bi = TextVectorizer("count", (1, 2)).fit(DOCS)
    assert len(bi.vocabulary_) > len(uni.vocabulary_) and "the cat" in bi.vocabulary_


def test_hash_encode_is_deterministic_bounded_and_ignores_oov():
    a = hash_encode("The cat sat", 100)
    assert a == hash_encode("the cat sat", 100)
    assert all(1 <= i < 100 for i in a)
    assert len(hash_encode("completely unseen words")) == 3


def test_bpe_round_trip_and_merges_shorten_sequences():
    corpus = ["lower lowest newer newest wider widest"] * 5
    bpe = BPE(num_merges=20).fit(corpus)
    text = "newest lower"
    assert bpe.decode(bpe.encode(text)) == text
    assert len(bpe.tokenize("newest")) < len("newest") + 1          # merged into subwords
    assert bpe.decode(bpe.encode("lowest")) == "lowest"


def test_bpe_unseen_word_is_split_into_known_pieces():
    bpe = BPE(num_merges=20).fit(["lower lowest newer newest"] * 5)
    assert bpe.decode(bpe.encode("wester")) == "wester"             # all characters are known


def test_toy_sentiment_classifier_learns():
    texts, y = make_toy_sentiment(400, seed=0)
    tp = TextProcess().fit(texts[:300])
    model, _ = train_embedding_classifier(texts[:300], y[:300], tp, epochs=25)
    X_test = tp.encode_batch(texts[300:], maxlen=8)
    acc = ((model.predict(X_test, verbose=0).ravel() >= 0.5) == y[300:]).mean()
    assert acc > 0.9
