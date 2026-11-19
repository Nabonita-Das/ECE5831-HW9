"""Bonus Part 1 experiments: augmentation ablation, seeds, size sweep, fine-tuning, ResNet50, error analysis.

Usage:  python run_bonus.py [--seeds 42 43 44] [--epochs 30] [--out bonus_results]
Results are cached in <out>/results.json, so an interrupted run can simply be restarted.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hw9 import experiments as X  # noqa: E402
from hw9.data import prepare_limited_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 150])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data-dir", default="limited_data")
    ap.add_argument("--raw-dir", default="data")
    ap.add_argument("--out", default="bonus_results")
    a = ap.parse_args()

    prepare_limited_dataset(a.data_dir, a.raw_dir)
    R = X.run_all(a.data_dir, a.out, a.raw_dir, tuple(a.seeds), tuple(a.sizes), a.epochs,
                  log=lambda m: print(m, flush=True))

    X.plot_ablation(R, save_path=f"{a.out}/ablation.png")
    X.plot_size_sweep(R, save_path=f"{a.out}/size_sweep.png")
    X.plot_final_comparison(R, save_path=f"{a.out}/final_comparison.png")
    npz = f"{a.out}/test_predictions.npz"
    X.plot_confusions(npz, save_path=f"{a.out}/confusion_matrices.png")
    X.plot_misclassified(npz, "vgg16_frozen", save_path=f"{a.out}/misclassified_vgg16.png")
    print("\n{:<34}{:>6}{:>11}{:>20}{:>10}".format("model", "runs", "train acc", "test acc", "test F1"))
    for r in X.summary_table(R):
        print("{:<34}{:>6}{:>11}{:>20}{:>10}".format(r["model"], r["runs"], r["train acc"], r["test acc"], r["test F1"]))


if __name__ == "__main__":
    main()
