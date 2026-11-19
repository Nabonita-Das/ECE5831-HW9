"""Part 1 pipeline: augmentation demo -> scratch CNN -> pretrained VGG16 -> comparison.

Usage:  python run_part1.py [--epochs-scratch 40] [--epochs-pre 30] [--out results]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tensorflow import keras  # noqa: E402

from hw9 import config as C  # noqa: E402
from hw9.augmentation import DataAug  # noqa: E402
from hw9.data import load_split, prepare_limited_dataset  # noqa: E402
from hw9.models import CatsDogsLimit, CatsDogsPre  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs-scratch", type=int, default=40)
    ap.add_argument("--epochs-pre", type=int, default=30)
    ap.add_argument("--data-dir", default="limited_data")
    ap.add_argument("--raw-dir", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    prepare_limited_dataset(args.data_dir, args.raw_dir)

    print("\n== DataAug ==")
    DataAug().show_dataset_samples(load_split(args.data_dir, "train", True),
                                   n_images=3, n_aug=5, save_prefix=f"{args.out}/augmentation")

    print("\n== CatsDogsLimit (from scratch) ==")
    scratch = CatsDogsLimit(args.data_dir, epochs=args.epochs_scratch)
    res_scratch = scratch.run()
    scratch.plot_history(f"{args.out}/scratch_history.png")

    print("\n== CatsDogsPre (pretrained VGG16) ==")
    pre = CatsDogsPre(args.data_dir, epochs=args.epochs_pre,
                      weights_path=os.path.join(args.raw_dir, "vgg16_notop.h5"))
    res_pre = pre.run()
    pre.plot_history(f"{args.out}/pretrained_history.png")

    results = {"scratch": res_scratch, "pretrained": res_pre}
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n== Comparison ==")
    print(f"{'model':<12}{'train acc':>10}{'val acc':>10}{'test acc':>10}")
    for name, r in results.items():
        print(f"{name:<12}{r['train']['accuracy']:>10.3f}{r['validation']['accuracy']:>10.3f}"
              f"{r['test']['accuracy']:>10.3f}")


if __name__ == "__main__":
    main()
