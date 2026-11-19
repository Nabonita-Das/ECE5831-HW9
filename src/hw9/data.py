"""Build the *limited* cats/dogs dataset on disk (train / validation / test)."""
import glob
import os
import random
import shutil
import urllib.request
import zipfile

from tensorflow import keras

from . import config as C


def download(url, dest):
    """Download url -> dest with a plain urllib request (browser-like header)."""
    print("Downloading", url)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _pool_from_official():
    zip_path = keras.utils.get_file("cats_and_dogs_filtered.zip", origin=C.OFFICIAL_URL, extract=True)
    root = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")
    pool = {}
    for c in C.CLASSES:
        files = []
        for split in ("train", "validation"):
            files += sorted(glob.glob(os.path.join(root, split, c, "*")))
        pool[c] = [f for f in files if f.lower().endswith(C.IMG_EXT)]
    return pool, "official cats_and_dogs_filtered (downloaded)"


def _pool_from_backup(raw_dir):
    pool = {}
    for c, z in C.BACKUP_ZIPS.items():
        zpath = os.path.join(raw_dir, z)
        if not os.path.exists(zpath):
            try:
                download(C.BACKUP_BASE + z, zpath)
            except Exception as e:
                raise FileNotFoundError(
                    f"{z} not found in '{raw_dir}' and could not be downloaded ({e}). "
                    f"Put CAT1.zip and DOG1.zip into '{raw_dir}'.") from e
        out = os.path.join(raw_dir, f"_raw_{c}")
        if not os.path.isdir(out):
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(out)
        pool[c] = sorted(f for f in glob.glob(os.path.join(out, "**", "*"), recursive=True)
                         if f.lower().endswith(C.IMG_EXT))
    return pool, "backup copy CAT1.zip / DOG1.zip (cat/dog photos)"


def prepare_limited_dataset(base="limited_data", raw_dir="data", seed=C.SEED, use_official=True):
    """Create base/{train,validation,test}/{cats,dogs} with N_TRAIN / N_VAL / N_TEST images per class.

    Both models always train on exactly the same images because the split is
    made with a fixed seed.
    """
    need = {"train": C.N_TRAIN, "validation": C.N_VAL, "test": C.N_TEST}
    ready = all(os.path.isdir(f"{base}/{s}/{c}") and len(os.listdir(f"{base}/{s}/{c}")) == n
                for s, n in need.items() for c in C.CLASSES)
    if ready:
        print("Limited dataset already prepared in:", base)
        return base

    pool = source = None
    if use_official:
        try:
            pool, source = _pool_from_official()
        except Exception as e:
            print(f"Official download not available ({type(e).__name__}) -> using backup copy")
    if pool is None:
        pool, source = _pool_from_backup(raw_dir)
    print("Data source:", source)

    rng = random.Random(seed)
    if os.path.isdir(base):
        shutil.rmtree(base)
    for c in C.CLASSES:
        files = list(pool[c])
        rng.shuffle(files)
        assert len(files) >= sum(need.values()), f"not enough {c} images: {len(files)}"
        start = 0
        for split, n in need.items():
            dst = f"{base}/{split}/{c}"
            os.makedirs(dst)
            for i, f in enumerate(files[start:start + n]):
                shutil.copy(f, os.path.join(dst, f"{c[:-1]}.{start + i}.jpg"))
            start += n
    return base


def load_split(base, split, shuffle):
    """tf.data pipeline for one split (binary labels: cats=0, dogs=1)."""
    return keras.utils.image_dataset_from_directory(
        f"{base}/{split}", image_size=C.IMG_SIZE, batch_size=C.BATCH,
        label_mode="binary", shuffle=shuffle, seed=C.SEED)
