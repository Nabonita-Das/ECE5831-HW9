import numpy as np

from hw9.augmentation import DataAug


def test_augmentation_keeps_shape_and_changes_pixels():
    imgs = np.random.RandomState(0).uniform(0, 255, (4, 150, 150, 3)).astype("float32")
    out = DataAug().augment(imgs).numpy()
    assert out.shape == imgs.shape
    assert not np.allclose(out, imgs)


def test_layer_is_identity_at_inference():
    imgs = np.random.RandomState(1).uniform(0, 255, (2, 150, 150, 3)).astype("float32")
    out = DataAug().get_layer()(imgs, training=False).numpy()
    assert np.allclose(out, imgs)
