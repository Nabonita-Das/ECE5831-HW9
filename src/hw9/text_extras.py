"""Extra text representations that build on TextProcess.

* TextVectorizer  - bag-of-words (count / binary) and TF-IDF vectors, optional n-grams
* hash_encode     - the hashing trick (fixed-size ids, no vocabulary, NOT reversible)
* BPE             - byte-pair-encoding subwords (reversible, no <OOV> for known characters)
* toy sentiment data + a small Embedding classifier showing the numbers feeding a real network
"""
import hashlib
import random
import re
from collections import Counter

import numpy as np

from .text_process import TextProcess


# ----------------------------------------------------------------------------- bag-of-words / TF-IDF
class TextVectorizer:
    """Turn documents into fixed-length count, binary or TF-IDF vectors."""

    def __init__(self, mode="tfidf", ngram_range=(1, 1)):
        assert mode in ("count", "binary", "tfidf")
        self.mode, self.ngram_range = mode, ngram_range
        self.tp = TextProcess(mode="word")
        self.vocabulary_, self.idf_ = {}, None

    def _terms(self, text):
        toks = [t for t in self.tp.tokenize(text) if re.match(r"\w", t)]      # drop punctuation tokens
        lo, hi = self.ngram_range
        return [" ".join(toks[i:i + n]) for n in range(lo, hi + 1) for i in range(len(toks) - n + 1)]

    def fit(self, docs):
        df = Counter(t for d in docs for t in set(self._terms(d)))
        self.vocabulary_ = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        self.idf_ = np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in self.vocabulary_], dtype="float32")
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocabulary_)), dtype="float32")
        for r, d in enumerate(docs):
            for t, c in Counter(self._terms(d)).items():
                if t in self.vocabulary_:
                    X[r, self.vocabulary_[t]] = c
        if self.mode == "binary":
            X = (X > 0).astype("float32")
        elif self.mode == "tfidf":
            X = X * self.idf_
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.where(norms == 0, 1, norms)
        return X

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def feature_names(self):
        return [t for t, _ in sorted(self.vocabulary_.items(), key=lambda kv: kv[1])]


# ----------------------------------------------------------------------------- hashing trick
def hash_encode(text, num_buckets=1000, mode="word"):
    """Map every token to md5(token) % num_buckets (id 0 is kept free for padding).

    No vocabulary is needed and unseen words never fail, but different words can collide and the
    mapping cannot be decoded back to text.
    """
    tp = TextProcess(mode=mode)
    return [int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % (num_buckets - 1) + 1 for t in tp.tokenize(text)]


# ----------------------------------------------------------------------------- byte-pair encoding
class BPE:
    """Small byte-pair-encoding tokenizer (words are split on white space, lower-cased).

    Frequent character pairs are merged into subwords, so rare / unseen words are split into known
    pieces instead of becoming <OOV>. Encoding and decoding are lossless for known characters.
    """

    END, UNK = "</w>", "<unk>"

    def __init__(self, num_merges=50):
        self.num_merges = num_merges
        self.merges, self.vocab, self.inv = [], {}, {}

    @staticmethod
    def _pairs(word):
        return Counter(zip(word, word[1:]))

    @staticmethod
    def _merge_word(word, pair):
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1]); i += 2
            else:
                out.append(word[i]); i += 1
        return tuple(out)

    def _split(self, word):
        return tuple(word) + (self.END,)

    def fit(self, texts):
        freq = Counter(w for t in texts for w in t.lower().split())
        words = {self._split(w): c for w, c in freq.items()}
        self.merges = []
        for _ in range(self.num_merges):
            pairs = Counter()
            for w, c in words.items():
                for p, k in self._pairs(w).items():
                    pairs[p] += k * c
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], p))
            self.merges.append(best)
            words = {self._merge_word(w, best): c for w, c in words.items()}
        symbols = {self.UNK} | {s for w in words for s in w} | {a + b for a, b in self.merges}   # every merge result
        chars = {ch for t in texts for ch in t.lower() if not ch.isspace()} | {self.END}
        self.vocab = {s: i + 1 for i, s in enumerate(sorted(symbols | chars))}      # 0 = padding
        self.inv = {i: s for s, i in self.vocab.items()}
        return self

    def tokenize_word(self, word):
        w = self._split(word.lower())
        for pair in self.merges:
            w = self._merge_word(w, pair)
        return list(w)

    def tokenize(self, text):
        return [p for word in text.lower().split() for p in self.tokenize_word(word)]

    def encode(self, text):
        return [self.vocab.get(p, self.vocab[self.UNK]) for p in self.tokenize(text)]

    def decode(self, ids):
        s = "".join(self.inv.get(int(i), self.UNK) for i in ids if int(i) != 0)
        return s.replace(self.END, " ").strip()


# ----------------------------------------------------------------------------- toy sentiment task
POSITIVE = ["great", "wonderful", "amazing", "fantastic", "excellent", "lovely", "brilliant", "enjoyable"]
NEGATIVE = ["terrible", "awful", "horrible", "boring", "disappointing", "poor", "dreadful", "annoying"]
NOUNS = ["movie", "film", "book", "meal", "service", "game", "trip", "show"]
TEMPLATES = ["The {n} was {a}.", "I thought the {n} was really {a}!", "What a {a} {n}.",
             "That {n} is {a}, honestly.", "Such a {a} {n} today."]


def make_toy_sentiment(n=400, seed=0):
    """Template-generated sentences (label 1 = positive, 0 = negative). A TOY dataset for demonstration only."""
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n):
        pos = rng.random() < 0.5
        adj = rng.choice(POSITIVE if pos else NEGATIVE)
        texts.append(rng.choice(TEMPLATES).format(n=rng.choice(NOUNS), a=adj))
        labels.append(int(pos))
    return texts, np.array(labels)


def train_embedding_classifier(train_texts, y_train, tp, maxlen=8, dim=8, epochs=30, seed=0, verbose=0):
    """Embedding -> average pooling -> sigmoid; consumes the integer ids produced by TextProcess."""
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(seed)
    X = tp.encode_batch(train_texts, maxlen=maxlen)
    model = keras.Sequential([
        keras.Input(shape=(maxlen,)),
        layers.Embedding(tp.vocab_size, dim, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(0.05), metrics=["accuracy"])
    hist = model.fit(X, y_train, epochs=epochs, batch_size=32, verbose=verbose)
    return model, hist
