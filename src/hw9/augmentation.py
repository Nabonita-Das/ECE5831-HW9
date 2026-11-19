"""DataAug: random image transformations that enlarge a small training set."""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


class DataAug:
    """Data augmentation helper (flip, rotation, zoom, translation, contrast)."""

    def __init__(self, seed=C.SEED):
        self.seed = seed
        self.augmentation = keras.Sequential([
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.15, seed=seed),
            layers.RandomZoom(0.2, seed=seed),
            layers.RandomTranslation(0.1, 0.1, seed=seed),
            layers.RandomContrast(0.1, seed=seed),
        ], name="data_augmentation")

    def get_layer(self):
        """The augmentation block, usable inside a model (active only while training)."""
        return self.augmentation

    def augment(self, images):
        """Augment a batch of images (training=True forces the random transforms on)."""
        return self.augmentation(images, training=True)

    def show_augmented(self, image, n=8, title="Augmented versions of one image", save_path=None):
        """Show the original image followed by n augmented versions of it."""
        image = tf.cast(image, tf.float32)
        batch = tf.repeat(tf.expand_dims(image, 0), n, axis=0)
        aug = self.augment(batch).numpy()
        fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
        axes[0].imshow(image.numpy().astype("uint8"))
        axes[0].set_title("original", fontsize=9)
        for i in range(n):
            axes[i + 1].imshow(np.clip(aug[i], 0, 255).astype("uint8"))
            axes[i + 1].set_title(f"aug {i + 1}", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(title)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        plt.show() if save_path is None else plt.close(fig)

    def show_dataset_samples(self, dataset, n_images=3, n_aug=5, save_prefix=None):
        """Take n_images originals from a tf.data dataset and show n_aug augmentations of each."""
        images, labels = next(iter(dataset.unbatch().batch(n_images)))
        for k, (img, lab) in enumerate(zip(images, labels)):
            name = dataset.class_names[int(lab.numpy().ravel()[0])]
            path = f"{save_prefix}_{k + 1}.png" if save_prefix else None
            self.show_augmented(img, n=n_aug, title=f"Original class: {name}", save_path=path)
