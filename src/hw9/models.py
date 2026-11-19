"""CatsDogsLimit (CNN from scratch) and CatsDogsPre (pretrained VGG16) on the same limited data."""
import contextlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C
from .augmentation import DataAug
from .data import download, load_split


def plot_history(history, title, save_path=None):
    """Training / validation accuracy and loss curves."""
    h = history.history
    ep = range(1, len(h["accuracy"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, k, name in ((ax[0], "accuracy", "accuracy"), (ax[1], "loss", "loss")):
        a.plot(ep, h[k], "o-", label="training")
        a.plot(ep, h["val_" + k], "s-", label="validation")
        a.set_title(f"{title} - {name}")
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


class CatsDogsLimit:
    """Cat/dog classifier trained FROM SCRATCH on the limited dataset (with data augmentation)."""

    def __init__(self, data_dir="limited_data", epochs=40, augment=True, seed=C.SEED, patience=10):
        self.data_dir, self.epochs, self.augment = data_dir, epochs, augment
        self.seed, self.patience = seed, patience
        self.model = self.history = None
        self.train_ds = self.val_ds = self.test_ds = None

    def load_data(self):
        self.train_ds = load_split(self.data_dir, "train", True, self.seed)
        self.val_ds = load_split(self.data_dir, "validation", False)
        self.test_ds = load_split(self.data_dir, "test", False)
        return self

    def build_model(self):
        keras.utils.set_random_seed(self.seed)
        inputs = keras.Input(shape=C.IMG_SIZE + (3,))
        x = layers.Rescaling(1. / 255)(inputs)
        if self.augment:
            x = DataAug(self.seed).get_layer()(x)
        for f in (32, 64, 128, 128):
            x = layers.Conv2D(f, 3, activation="relu")(x)
            x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs, name="cats_dogs_scratch")
        self.model.compile(loss="binary_crossentropy",
                           optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"])
        return self.model

    def train(self, verbose=2):
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)]
        self.history = self.model.fit(self.train_ds, validation_data=self.val_ds,
                                      epochs=self.epochs, callbacks=cb, verbose=verbose)
        return self.history

    def evaluate(self):
        res = {}
        for name, ds in (("train", self.train_ds), ("validation", self.val_ds), ("test", self.test_ds)):
            loss, acc = self.model.evaluate(ds, verbose=0)
            res[name] = {"loss": float(loss), "accuracy": float(acc)}
        return res

    def predict_test(self):
        """Test-set probabilities, true labels and file paths (fixed order)."""
        probs = self.model.predict(self.test_ds, verbose=0).ravel()
        labels = np.concatenate([y.numpy().ravel() for _, y in self.test_ds])
        return probs, labels, list(self.test_ds.file_paths)

    def plot_history(self, save_path=None):
        plot_history(self.history, "Scratch CNN (limited data)", save_path)

    def run(self, verbose=2):
        self.load_data()
        self.build_model()
        self.train(verbose)
        return self.evaluate()


class CatsDogsPre:
    """Cat/dog classifier using a frozen, pretrained VGG16 base (feature extraction) + new head."""

    LOCAL_WEIGHTS = "vgg16_notop.h5"

    def __init__(self, data_dir="limited_data", epochs=30, weights_path=None):
        self.data_dir, self.epochs = data_dir, epochs
        self.weights_path = weights_path or self.LOCAL_WEIGHTS
        self.base = self.model = self.history = None
        self.feats, self.labels = {}, {}

    def load_pretrained_base(self):
        shape = C.IMG_SIZE + (3,)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=shape)
            print("Loaded VGG16 ImageNet weights (Keras download).")
        except Exception:
            if not os.path.exists(self.weights_path):
                download(C.VGG16_NOTOP_URL, self.weights_path)       # ~58 MB, one time
            self.base = keras.applications.VGG16(weights=self.weights_path, include_top=False, input_shape=shape)
            print("Loaded VGG16 ImageNet weights from", self.weights_path)
        self.base.trainable = False
        return self.base

    def extract_features(self):
        """Pass every split through the frozen base ONCE and cache the 4x4x512 feature maps."""
        for split in ("train", "validation", "test"):
            ds = load_split(self.data_dir, split, shuffle=False)      # fixed order keeps labels aligned
            fs, ls = [], []
            for x, y in ds:
                fs.append(self.base.predict(keras.applications.vgg16.preprocess_input(x.numpy()), verbose=0))
                ls.append(y.numpy().ravel())
            self.feats[split], self.labels[split] = np.concatenate(fs), np.concatenate(ls)

    def build_model(self):
        keras.utils.set_random_seed(C.SEED)
        inputs = keras.Input(shape=self.feats["train"].shape[1:])
        x = layers.Flatten()(inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(inputs, outputs, name="cats_dogs_pretrained_head")
        self.model.compile(loss="binary_crossentropy",
                           optimizer=keras.optimizers.RMSprop(2e-5), metrics=["accuracy"])
        return self.model

    def train(self, verbose=2):
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
        self.history = self.model.fit(
            self.feats["train"], self.labels["train"],
            validation_data=(self.feats["validation"], self.labels["validation"]),
            epochs=self.epochs, batch_size=C.BATCH, callbacks=cb, verbose=verbose)
        return self.history

    def evaluate(self):
        res = {}
        for split in ("train", "validation", "test"):
            loss, acc = self.model.evaluate(self.feats[split], self.labels[split], verbose=0)
            res[split] = {"loss": float(loss), "accuracy": float(acc)}
        return res

    def plot_history(self, save_path=None):
        plot_history(self.history, "Pretrained VGG16 + new head (limited data)", save_path)

    # ---- Variant B: frozen VGG16 + DataAug, trained end-to-end on the images -------------------
    def build_augmented_model(self):
        """raw image -> DataAug (training only) -> VGG16 preprocessing -> frozen VGG16 base -> new head."""
        keras.utils.set_random_seed(C.SEED)
        inputs = keras.Input(shape=C.IMG_SIZE + (3,))
        x = DataAug().get_layer()(inputs)
        x = keras.applications.vgg16.preprocess_input(x)
        x = self.base(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        self.aug_model = keras.Model(inputs, outputs, name="cats_dogs_pretrained_augmented")
        self.aug_model.compile(loss="binary_crossentropy",
                               optimizer=keras.optimizers.RMSprop(1e-4), metrics=["accuracy"])
        return self.aug_model

    def train_augmented(self, epochs=12, verbose=2):
        self.aug_train_ds = load_split(self.data_dir, "train", True)
        self.aug_val_ds = load_split(self.data_dir, "validation", False)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        self.aug_history = self.aug_model.fit(self.aug_train_ds, validation_data=self.aug_val_ds,
                                              epochs=epochs, callbacks=cb, verbose=verbose)
        return self.aug_history

    def evaluate_augmented(self):
        res = {}
        for name, ds in (("train", load_split(self.data_dir, "train", False)), ("validation", self.aug_val_ds),
                         ("test", load_split(self.data_dir, "test", False))):
            loss, acc = self.aug_model.evaluate(ds, verbose=0)
            res[name] = {"loss": float(loss), "accuracy": float(acc)}
        return res

    def build_full_model(self):
        """Assemble base + trained head into one end-to-end model that accepts raw images."""
        inputs = keras.Input(shape=C.IMG_SIZE + (3,))
        x = keras.applications.vgg16.preprocess_input(inputs)
        return keras.Model(inputs, self.model(self.base(x)), name="cats_dogs_pretrained_full")

    def run(self, verbose=2):
        self.load_pretrained_base()
        self.extract_features()
        self.build_model()
        self.train(verbose)
        return self.evaluate()
