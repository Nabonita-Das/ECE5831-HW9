# ECE-5831 HW#9 - Data Augmentation, Transfer Learning and Text Processing

This repository contains my solution to HW#9. It has two parts:

| Part | Topic | Deliverable |
|---|---|---|
| 1 | Cat/dog image classification with a **very small training set**: data augmentation, a CNN trained from scratch, and a pretrained VGG16 | `Module_9_1.ipynb` |
| 2 | Text processing: encoding text into numbers and decoding it back | `Module_9_2.ipynb` |

The two notebooks are the graded deliverables (all cells executed, outputs and markdown included).
The same code is also available as a small, tested Python package in `src/hw9`, with two command-line scripts.

---

## 1. Repository layout

```
.
├── Module_9_1.ipynb          # Part 1 deliverable (executed, with outputs)
├── Module_9_2.ipynb          # Part 2 deliverable (executed, with outputs)
├── run_part1.py              # Part 1 as a script: augmentation -> scratch CNN -> VGG16 -> comparison
├── run_part2.py              # Part 2 as a script: text -> numbers -> text demo
├── src/hw9/
│   ├── config.py             # all constants (image size, split sizes, URLs, seed)
│   ├── data.py               # builds the limited train/validation/test dataset
│   ├── augmentation.py       # class DataAug
│   ├── models.py             # classes CatsDogsLimit and CatsDogsPre
│   └── text_process.py       # class TextProcess
├── tests/                    # pytest unit tests (19 tests)
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## 2. Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                               # unit tests (a few seconds)
python run_part2.py                  # Part 2 demo
python run_part1.py                  # Part 1 (a few minutes on CPU, downloads data on first run)
jupyter notebook                     # or open the notebooks in VS Code
```

Tested with Python 3.13 and TensorFlow 2.21 (CPU). A GPU is not required.

---

## 3. Part 1 - image classification with limited data

### 3.1 The problem
A convolutional network needs a lot of labelled images. When only a few hundred are available, a model
trained from scratch **over-fits**: it memorises the training photos and does poorly on new ones.
Two standard remedies are compared here:

1. **Data augmentation** - create many slightly different versions of every training image.
2. **Transfer learning** - re-use a network that was already trained on a huge dataset (ImageNet).

### 3.2 The limited dataset (`data.py`)
A balanced dataset is built on disk from a pool of cat and dog photos, shuffled with a fixed seed:

| split | images per class | total | purpose |
|---|---|---|---|
| train (limited) | 150 | **300** | fitting the models |
| validation | 50 | 100 | monitoring training / early stopping |
| test | 100 | 200 | final, untouched evaluation |

Both models are trained on **exactly the same 300 images**, which makes the comparison fair.

Data sources, tried in this order:
1. the official `cats_and_dogs_filtered` dataset (Keras download);
2. a backup copy of cat/dog photos (`CAT1.zip`, `DOG1.zip`) downloaded from a public GitHub repository,
   used automatically when the official download is blocked (HTTP 403).

Downloaded files are stored in `data/` and `limited_data/`, which are git-ignored.

### 3.3 `DataAug` (`augmentation.py`)
A `keras.Sequential` block of random transformations:

| transformation | strength |
|---|---|
| horizontal flip | random |
| rotation | up to about 15 degrees |
| zoom | +/- 20 % |
| translation | +/- 10 % |
| contrast | +/- 10 % |

The block is only active during training (Keras random layers are the identity at inference), so it can
be placed **inside** the model without affecting validation or test accuracy.
`show_dataset_samples()` displays original images next to their augmented versions.

### 3.4 `CatsDogsLimit` - CNN from scratch (`models.py`)
```
Rescaling(1/255) -> DataAug -> [Conv(32) + MaxPool] -> [Conv(64) + MaxPool]
-> [Conv(128) + MaxPool] -> [Conv(128) + MaxPool] -> Flatten -> Dropout(0.5)
-> Dense(256, ReLU) -> Dense(1, sigmoid)
```
Binary cross-entropy loss, RMSprop (lr 1e-3), early stopping on validation loss (patience 10, best weights restored).

### 3.5 `CatsDogsPre` - pretrained VGG16 (`models.py`)
* VGG16 with ImageNet weights and `include_top=False` is used as a **frozen feature extractor**.
* Every image is passed through the frozen base once; the resulting 4x4x512 feature maps are cached.
  This is mathematically equivalent to running a frozen base in every epoch, but much faster on a CPU.
* Only a small new head is trained: `Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid)`
  (RMSprop, lr 2e-5, early stopping).
* `build_full_model()` assembles base + head into one end-to-end model that accepts raw images.
* Weights are loaded with `weights="imagenet"`; if that download is blocked they are fetched from GitHub
  and cached as `vgg16_notop.h5`.

### 3.6 Results (notebook run, same 300 training images)

| Model | train acc | validation acc | test acc |
|---|---|---|---|
| From scratch (CNN + augmentation) | 76.7 % | 71.0 % | **70.5 %** |
| Pretrained VGG16 (frozen base + new head) | 100 % | 93.0 % | **96.0 %** |

The pretrained model is about **25 percentage points** better on the held-out test set.
Exact numbers vary slightly between machines and library versions (random initialisation); the conclusion does not.

### 3.7 Discussion
* **Scratch.** The network must learn edges, textures, shapes and the "cat vs dog" concept from only 300 images.
  Augmentation and dropout help, but there is simply too little data, so accuracy on new images stays low and
  the validation curve is noisy.
* **Pretrained.** The VGG16 base already encodes general visual features learned from 1.4 million ImageNet images
  (which include many cat and dog breeds). Only a tiny classifier has to be learned, so training is fast and
  generalisation is much better.
* **Take-away.** With little labelled data, transfer learning is the most effective tool; augmentation is the cheap
  way to get more out of a model that has to be trained from scratch. They can be combined (augmentation +
  fine-tuning the last VGG block) for further gains.

---

## 4. Part 2 - text processing (`text_process.py`)

Neural networks need numbers, not strings. `TextProcess` converts text to numbers and back.

| Step | Method | What it does |
|---|---|---|
| tokenise | `tokenize` | words (punctuation is its own token, `don't` stays together) or single characters (`mode="char"`) |
| vocabulary | `fit(texts)` | most frequent token gets the smallest id; `0 = <PAD>`, `1 = <OOV>` are reserved |
| **encode** | `encode(text)` | text -> list of integer ids (unknown words -> `<OOV>`) |
| | `encode_batch(texts, maxlen)` | list of texts -> padded / truncated 2-D integer array |
| | `one_hot_encode(text)` | text -> `(tokens, vocab_size)` one-hot matrix |
| **decode** | `decode(ids)` | ids -> readable text (padding skipped, no space before punctuation) |
| | `decode_batch(array)` | 2-D array -> list of texts |
| | `one_hot_decode(matrix)` | `argmax` per row, then decode |
| persist | `to_json()` / `from_json()` | save and restore the vocabulary |

```python
from hw9 import TextProcess

tp = TextProcess().fit(["The cat sat on the mat.", "The dog chased the cat!"])
ids = tp.encode("The dog chased the cat!")      # [2, 8, 9, 2, 3, 10]
tp.decode(ids)                                  # 'the dog chased the cat!'
```

Properties (all covered by tests):
* word-level round trip is exact up to lower-casing and whitespace normalisation;
* words that were not seen in `fit` become `<OOV>` and cannot be recovered;
* character-level mode is fully lossless;
* the notebook also cross-checks the idea against Keras `TextVectorization`.

---

## 5. Tests

`pytest` runs 19 tests:
* `tests/test_text_process.py` - vocabulary, encoding, decoding, padding, one-hot, OOV, char mode, JSON.
* `tests/test_augmentation.py` - augmentation keeps the image shape, changes the pixels while training and is the identity at inference.

## 6. Design notes and limitations
* Seeds are fixed (`config.SEED = 42`) for reproducibility on the same setup.
* The "limited" dataset is deliberately tiny (300 training images) to show the effect of transfer learning.
* The pretrained model is trained as feature extraction without augmentation and without fine-tuning; adding
  both is a natural extension.
* Test images are never used for training, early stopping or model selection.
