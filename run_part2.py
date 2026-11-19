"""Part 2 demo: text -> numbers -> text with TextProcess.

Usage:  python run_part2.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hw9.text_process import TextProcess  # noqa: E402

CORPUS = [
    "The cat sat on the mat.",
    "The dog chased the cat!",
    "Dogs and cats don't always get along.",
    "Is the dog happy? Yes, the dog is very happy.",
]


def main():
    tp = TextProcess(mode="word").fit(CORPUS)
    print("vocabulary size:", tp.vocab_size)
    for s in CORPUS:
        ids = tp.encode(s)
        print(f"{s!r}\n  ids : {ids}\n  back: {tp.decode(ids)!r}")
    batch = tp.encode_batch(["The cat sat on the mat.", "Is the dog happy?"])
    print("padded batch:\n", batch, "\ndecoded:", tp.decode_batch(batch))
    oh = tp.one_hot_encode("the dog is happy")
    print("one-hot shape:", oh.shape, "-> decoded:", tp.one_hot_decode(oh))
    print("unknown word:", tp.decode(tp.encode("The elephant sat on the mat.")))


if __name__ == "__main__":
    main()
