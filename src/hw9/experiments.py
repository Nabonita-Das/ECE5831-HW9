"""Extra experiments for Part 1 (all on the SAME limited cats/dogs data).

* augmentation ablation (scratch CNN with vs. without DataAug)
* repeated runs with several seeds (mean +/- std)
* training-set-size sweep (50 / 100 / 150 images per class)
* pretrained VGG16: frozen feature extraction vs. fine-tuning the last block
* a second pretrained backbone (ResNet50)
* error analysis (confusion matrices, precision / recall, misclassified images)
"""
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C
from .data import download, load_split
from .models import CatsDogsLimit

RESNET50_NOTOP_URL = ("https://github.com/keras-team/keras-applications/releases/download/"
                      "resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
SHAPE = C.IMG_SIZE + (3,)


# ----------------------------------------------------------------------------- data helpers
def make_subset(base, n_per_class, out=None):
    """Copy the first n training images per class (sorted by name) + the full val/test sets."""
    out = out or f"{base}_n{n_per_class}"
    if os.path.isdir(out):
        shutil.rmtree(out)
    for c in C.CLASSES:
        src = f"{base}/train/{c}"
        dst = f"{out}/train/{c}"
        os.makedirs(dst)
        for f in sorted(os.listdir(src))[:n_per_class]:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        for split in ("validation", "test"):
            shutil.copytree(f"{base}/{split}/{c}", f"{out}/{split}/{c}")
    return out


def _metrics(probs, labels):
    pred = (np.asarray(probs) >= 0.5).astype(int)
    y = np.asarray(labels).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


# ----------------------------------------------------------------------------- backbones / features
def load_backbone(name, weights_dir="data", layer=None):
    """Frozen ImageNet backbone (no top). `layer` truncates the network at that layer."""
    os.makedirs(weights_dir, exist_ok=True)
    if name == "vgg16":
        ctor, url, fname = keras.applications.VGG16, C.VGG16_NOTOP_URL, "vgg16_notop.h5"
    elif name == "resnet50":
        ctor, url, fname = keras.applications.ResNet50, RESNET50_NOTOP_URL, "resnet50_notop.h5"
    else:
        raise ValueError(name)
    try:
        base = ctor(weights="imagenet", include_top=False, input_shape=SHAPE)
    except Exception:
        path = os.path.join(weights_dir, fname)
        if not os.path.exists(path):
            download(url, path)
        base = ctor(weights=path, include_top=False, input_shape=SHAPE)
    base.trainable = False
    if layer:
        base = keras.Model(base.input, base.get_layer(layer).output)
    return base


def _preprocess(name):
    return {"vgg16": keras.applications.vgg16.preprocess_input,
            "resnet50": keras.applications.resnet50.preprocess_input}[name]


def extract_features(data_dir, name="vgg16", layer=None, weights_dir="data"):
    """Run every split through the frozen backbone once -> {split: (X, y, file_paths)}."""
    base = load_backbone(name, weights_dir, layer)
    pre = _preprocess(name)
    out = {}
    for split in ("train", "validation", "test"):
        ds = load_split(data_dir, split, shuffle=False)
        fs, ls = [], []
        for x, y in ds:
            fs.append(base.predict(pre(x.numpy()), verbose=0))
            ls.append(y.numpy().ravel())
        out[split] = (np.concatenate(fs), np.concatenate(ls), list(ds.file_paths))
    return out


def subset_indices(labels, n_per_class):
    """Indices of the first n examples of each class (features are stored in sorted file order)."""
    return np.concatenate([np.where(labels == c)[0][:n_per_class] for c in (0, 1)])


# ----------------------------------------------------------------------------- models on cached features
def _fit(model, Xtr, ytr, Xva, yva, epochs, verbose=0):
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    return model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                     batch_size=C.BATCH, callbacks=cb, verbose=verbose)


def _head(x, units=256):
    x = layers.Flatten()(x)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return layers.Dense(1, activation="sigmoid")(x)


def _result(model, feats, probs_test):
    res = {}
    for split in ("train", "validation", "test"):
        X, y, _ = feats[split]
        res[split] = {"accuracy": float(model.evaluate(X, y, verbose=0)[1])}
    res["test_metrics"] = _metrics(probs_test, feats["test"][1])
    return res


def train_frozen_head(feats, n_per_class, seed, lr=2e-5, epochs=30):
    """Classifier head on frozen backbone features (VGG16 block5 / ResNet50)."""
    keras.utils.set_random_seed(seed)
    Xtr, ytr, _ = feats["train"]
    idx = subset_indices(ytr, n_per_class)
    inp = keras.Input(shape=Xtr.shape[1:])
    model = keras.Model(inp, _head(inp))
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(lr), metrics=["accuracy"])
    _fit(model, Xtr[idx], ytr[idx], feats["validation"][0], feats["validation"][1], epochs)
    probs = model.predict(feats["test"][0], verbose=0).ravel()
    res = {}
    for split in ("validation", "test"):
        res[split] = {"accuracy": float(model.evaluate(feats[split][0], feats[split][1], verbose=0)[1])}
    res["train"] = {"accuracy": float(model.evaluate(Xtr[idx], ytr[idx], verbose=0)[1])}
    res["test_metrics"] = _metrics(probs, feats["test"][1])
    return res, probs


def train_finetuned_vgg(feats_block4, n_per_class, seed, weights_dir="data", lr=1e-5, epochs=15):
    """Fine-tune VGG16's last conv block (block5) + new head, starting from cached block4 features."""
    keras.utils.set_random_seed(seed)
    vgg = keras.applications.VGG16(weights=os.path.join(weights_dir, "vgg16_notop.h5"),
                                   include_top=False, input_shape=SHAPE) \
        if os.path.exists(os.path.join(weights_dir, "vgg16_notop.h5")) else \
        keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=SHAPE)
    names = [l.name for l in vgg.layers]
    Xtr, ytr, _ = feats_block4["train"]
    idx = subset_indices(ytr, n_per_class)
    inp = keras.Input(shape=Xtr.shape[1:])
    x = inp
    for layer in vgg.layers[names.index("block5_conv1"):]:
        layer.trainable = True
        x = layer(x)
    model = keras.Model(inp, _head(x))
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(lr), metrics=["accuracy"])
    _fit(model, Xtr[idx], ytr[idx], feats_block4["validation"][0], feats_block4["validation"][1], epochs)
    probs = model.predict(feats_block4["test"][0], verbose=0).ravel()
    res = {}
    for split in ("validation", "test"):
        res[split] = {"accuracy": float(model.evaluate(feats_block4[split][0], feats_block4[split][1], verbose=0)[1])}
    res["train"] = {"accuracy": float(model.evaluate(Xtr[idx], ytr[idx], verbose=0)[1])}
    res["test_metrics"] = _metrics(probs, feats_block4["test"][1])
    return res, probs


def train_scratch(data_dir, augment, seed, epochs=30):
    m = CatsDogsLimit(data_dir, epochs=epochs, augment=augment, seed=seed, patience=8)
    m.load_data(); m.build_model(); m.train(verbose=0)
    res = m.evaluate()
    probs, labels, paths = m.predict_test()
    res["test_metrics"] = _metrics(probs, labels)
    keras.backend.clear_session()
    return {k: ({"accuracy": v["accuracy"]} if k in ("train", "validation", "test") else v)
            for k, v in res.items()}, probs, labels, paths


# ----------------------------------------------------------------------------- orchestration
def run_all(data_dir="limited_data", out_dir="bonus_results", weights_dir="data",
            seeds=(42, 43, 44), sizes=(50, 100, 150), epochs=30, log=print):
    """Run every experiment; results are written incrementally to out_dir/results.json."""
    os.makedirs(out_dir, exist_ok=True)
    rpath = f"{out_dir}/results.json"
    R = json.load(open(rpath)) if os.path.exists(rpath) else {}
    preds = {}
    full = C.N_TRAIN                      # the main limited dataset (150 images per class)
    assert max(sizes) <= full

    def save():
        json.dump(R, open(rpath, "w"), indent=1)

    def put(model, n, seed, res):
        R.setdefault(model, {}).setdefault(str(n), {})[str(seed)] = res
        save()

    def done(model, n, seed):
        return str(seed) in R.get(model, {}).get(str(n), {})

    # ---- pretrained features (cached once)
    log("extracting VGG16 block5 features ...")
    f_vgg = extract_features(data_dir, "vgg16", None, weights_dir)
    log("extracting VGG16 block4 features ...")
    f_b4 = extract_features(data_dir, "vgg16", "block4_pool", weights_dir)
    log("extracting ResNet50 features ...")
    f_res = extract_features(data_dir, "resnet50", None, weights_dir)
    labels_test = f_vgg["test"][1]
    test_paths = f_vgg["test"][2]

    for n in sizes:
        for s in seeds:
            if not done("vgg16_frozen", n, s):
                res, p = train_frozen_head(f_vgg, n, s)
                put("vgg16_frozen", n, s, res)
                if n == full and s == seeds[0]:
                    preds["vgg16_frozen"] = p
                log(f"vgg16_frozen n={n} seed={s} test={res['test']['accuracy']:.3f}")
            if not done("resnet50_frozen", n, s):
                res, p = train_frozen_head(f_res, n, s, lr=1e-4)
                put("resnet50_frozen", n, s, res)
                if n == full and s == seeds[0]:
                    preds["resnet50_frozen"] = p
                log(f"resnet50_frozen n={n} seed={s} test={res['test']['accuracy']:.3f}")

    for s in seeds:
        if not done("vgg16_finetune", full, s):
            res, p = train_finetuned_vgg(f_b4, full, s, weights_dir)
            put("vgg16_finetune", full, s, res)
            if s == seeds[0]:
                preds["vgg16_finetune"] = p
            log(f"vgg16_finetune n={full} seed={s} test={res['test']['accuracy']:.3f}")

    # ---- scratch CNN: augmentation ablation at full size, sweep for the smaller sizes
    for n in sorted(sizes, reverse=True):
        d = data_dir if n == full else make_subset(data_dir, n)
        for augment, name in ((True, "scratch_aug"), (False, "scratch_noaug")):
            if name == "scratch_noaug" and n != full:
                continue
            for s in seeds:
                if done(name, n, s):
                    continue
                res, p, y, paths = train_scratch(d, augment, s, epochs)
                put(name, n, s, res)
                if n == full and s == seeds[0]:
                    preds[name] = p
                log(f"{name} n={n} seed={s} test={res['test']['accuracy']:.3f}")
        if n != full:
            shutil.rmtree(d, ignore_errors=True)

    if preds:
        np.savez(f"{out_dir}/test_predictions.npz", labels=labels_test,
                 paths=np.array(test_paths), **preds)
    return R


# ----------------------------------------------------------------------------- summaries & plots
LABELS = {"scratch_noaug": "Scratch CNN\n(no augmentation)", "scratch_aug": "Scratch CNN\n(+ augmentation)",
          "vgg16_frozen": "VGG16\n(frozen)", "vgg16_finetune": "VGG16\n(fine-tuned block5)",
          "resnet50_frozen": "ResNet50\n(frozen)"}


def acc(R, model, n, split="test"):
    """Array of accuracies over seeds."""
    return np.array([r[split]["accuracy"] for r in R[model][str(n)].values()])


def summary_table(R, n=150):
    rows = []
    for m in ("scratch_noaug", "scratch_aug", "vgg16_frozen", "vgg16_finetune", "resnet50_frozen"):
        if m in R and str(n) in R[m]:
            tr, te = acc(R, m, n, "train"), acc(R, m, n, "test")
            f1 = np.mean([r["test_metrics"]["f1"] for r in R[m][str(n)].values()])
            rows.append({"model": LABELS[m].replace("\n", " "), "runs": len(te),
                         "train acc": f"{tr.mean():.3f}", "test acc": f"{te.mean():.3f} +/- {te.std():.3f}",
                         "test F1": f"{f1:.3f}"})
    return rows


def _bar_errors(ax, names, series, title):
    means = [s.mean() for s in series]; stds = [s.std() for s in series]
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color="#4c78a8")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("test accuracy"); ax.set_title(title); ax.grid(axis="y", alpha=.3)
    for b, m, sd in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + sd + 0.012, f"{m:.1%}", ha="center", fontsize=8)


def plot_ablation(R, n=150, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ms = ["scratch_noaug", "scratch_aug"]
    _bar_errors(ax, [LABELS[m] for m in ms], [acc(R, m, n) for m in ms],
                f"Effect of data augmentation (n={n}/class)")
    plt.tight_layout(); _save(fig, save_path)


def plot_final_comparison(R, n=150, save_path=None):
    ms = [m for m in ("scratch_noaug", "scratch_aug", "vgg16_frozen", "vgg16_finetune", "resnet50_frozen")
          if m in R and str(n) in R[m]]
    fig, ax = plt.subplots(figsize=(9, 4))
    _bar_errors(ax, [LABELS[m] for m in ms], [acc(R, m, n) for m in ms],
                f"All models, same {2 * n} training images (mean +/- std over seeds)")
    plt.tight_layout(); _save(fig, save_path)


def plot_size_sweep(R, save_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, color in (("scratch_aug", "#e45756"), ("vgg16_frozen", "#4c78a8"), ("resnet50_frozen", "#54a24b")):
        sizes = sorted(int(k) for k in R[m])
        means = [acc(R, m, n).mean() for n in sizes]; stds = [acc(R, m, n).std() for n in sizes]
        ax.errorbar([2 * n for n in sizes], means, yerr=stds, marker="o", capsize=4, label=LABELS[m].replace("\n", " "), color=color)
    ax.set_xlabel("training images (total)"); ax.set_ylabel("test accuracy"); ax.set_ylim(0.4, 1.02)
    ax.set_title("Accuracy vs. training-set size"); ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); _save(fig, save_path)


def plot_confusions(npz_path, models=("scratch_aug", "vgg16_frozen", "vgg16_finetune"), save_path=None):
    d = np.load(npz_path, allow_pickle=True)
    y = d["labels"].astype(int)
    models = [m for m in models if m in d.files]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        pred = (d[m] >= 0.5).astype(int)
        cm = np.array([[((y == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["cat", "dog"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["cat", "dog"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(LABELS[m].replace("\n", " "), fontsize=9)
    plt.tight_layout(); _save(fig, save_path)


def plot_misclassified(npz_path, model="vgg16_frozen", k=8, save_path=None):
    d = np.load(npz_path, allow_pickle=True)
    y = d["labels"].astype(int); p = d[model]; paths = d["paths"]
    wrong = np.where((p >= 0.5).astype(int) != y)[0]
    wrong = wrong[np.argsort(-np.abs(p[wrong] - 0.5))][:k]          # most confident mistakes first
    if len(wrong) == 0:
        print("no misclassified test images"); return
    cols = min(4, len(wrong)); rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.1 * rows)); axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, wrong):
        ax.imshow(keras.utils.load_img(paths[i], target_size=C.IMG_SIZE)); ax.axis("off")
        ax.set_title(f"true {C.CLASSES[y[i]][:-1]}, P(dog)={p[i]:.2f}", fontsize=8)
    for ax in axes[len(wrong):]:
        ax.axis("off")
    fig.suptitle(f"Most confident mistakes - {LABELS[model].replace(chr(10), ' ')}")
    plt.tight_layout(); _save(fig, save_path)


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=130); plt.close(fig)
    else:
        plt.show()
