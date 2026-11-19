"""TextProcess: encode text into numbers (ids, padded batches, one-hot) and decode it back."""
import json
import re
from collections import Counter

import numpy as np


class TextProcess:
    """Encode text into integer / one-hot representations and decode it back."""

    PAD, OOV = "<PAD>", "<OOV>"
    TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")      # words (keeping don't), or one punctuation mark

    def __init__(self, mode="word", lower=True, num_words=None):
        assert mode in ("word", "char"), "mode must be 'word' or 'char'"
        self.mode, self.lower, self.num_words = mode, lower, num_words
        self.word_index = {self.PAD: 0, self.OOV: 1}      # token -> id
        self.index_word = {0: self.PAD, 1: self.OOV}      # id -> token
        self.word_counts = Counter()

    # ---------- preprocessing ----------
    def normalize(self, text):
        """Lower-case (optional) and collapse repeated white space."""
        text = text.lower() if self.lower else text
        return re.sub(r"\s+", " ", text).strip() if self.mode == "word" else text

    def tokenize(self, text):
        text = self.normalize(text)
        return self.TOKEN_RE.findall(text) if self.mode == "word" else list(text)

    # ---------- vocabulary ----------
    def fit(self, texts):
        """Build the vocabulary from a list of texts (most frequent token -> smallest id)."""
        self.word_counts = Counter(tok for t in texts for tok in self.tokenize(t))
        most_common = self.word_counts.most_common(self.num_words)
        self.word_index = {self.PAD: 0, self.OOV: 1}
        for tok, _ in most_common:
            self.word_index.setdefault(tok, len(self.word_index))
        self.index_word = {i: t for t, i in self.word_index.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.word_index)

    # ---------- ENCODING ----------
    def encode(self, text):
        """text -> list of integer ids (unknown tokens -> OOV id)."""
        oov = self.word_index[self.OOV]
        return [self.word_index.get(tok, oov) for tok in self.tokenize(text)]

    def encode_batch(self, texts, maxlen=None, padding="post"):
        """list of texts -> 2-D numpy array of ids, padded (with 0) / truncated to maxlen."""
        seqs = [self.encode(t) for t in texts]
        maxlen = maxlen or max(len(s) for s in seqs)
        out = np.zeros((len(seqs), maxlen), dtype="int32")
        for i, s in enumerate(seqs):
            s = s[:maxlen]
            if padding == "post":
                out[i, :len(s)] = s
            else:
                out[i, maxlen - len(s):] = s
        return out

    def one_hot_encode(self, text_or_ids):
        """text (or list of ids) -> (sequence_length, vocab_size) one-hot matrix."""
        ids = self.encode(text_or_ids) if isinstance(text_or_ids, str) else list(text_or_ids)
        m = np.zeros((len(ids), self.vocab_size), dtype="float32")
        m[np.arange(len(ids)), ids] = 1.0
        return m

    # ---------- DECODING ----------
    def _join(self, tokens):
        if self.mode == "char":
            return "".join(tokens)
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:)\]%])", r"\1", text)    # no space before closing punctuation
        text = re.sub(r"([(\[])\s+", r"\1", text)            # no space after opening bracket
        return text

    def decode(self, ids, skip_padding=True):
        """list of ids -> readable text."""
        toks = [self.index_word.get(int(i), self.OOV) for i in ids]
        if skip_padding:
            toks = [t for t in toks if t != self.PAD]
        return self._join(toks)

    def decode_batch(self, array):
        """2-D array of ids -> list of texts (padding removed)."""
        return [self.decode(row) for row in array]

    def one_hot_decode(self, matrix):
        """one-hot matrix (or probabilities) -> text, using argmax on every row."""
        return self.decode(np.argmax(matrix, axis=1))

    # ---------- utilities ----------
    def to_json(self, path=None):
        d = {"mode": self.mode, "lower": self.lower, "word_index": self.word_index}
        s = json.dumps(d, indent=1)
        if path:
            open(path, "w").write(s)
        return s

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        tp = cls(mode=d["mode"], lower=d["lower"])
        tp.word_index = d["word_index"]
        tp.index_word = {i: t for t, i in tp.word_index.items()}
        return tp
