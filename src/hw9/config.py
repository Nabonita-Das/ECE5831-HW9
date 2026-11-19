"""Central configuration shared by every module (change values here, not in the code)."""

SEED = 42

# image pipeline
IMG_SIZE = (150, 150)          # every image is resized to 150 x 150
BATCH = 20
CLASSES = ["cats", "dogs"]
IMG_EXT = (".jpg", ".jpeg", ".png")

# limited dataset: images PER CLASS
N_TRAIN, N_VAL, N_TEST = 150, 50, 100    # -> only 300 training images in total

# data sources
OFFICIAL_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
BACKUP_BASE = ("https://raw.githubusercontent.com/samialarabiyat/"
               "Cat-vs-Dog-Classification-with-Transfer-Learning/main/")
BACKUP_ZIPS = {"cats": "CAT1.zip", "dogs": "DOG1.zip"}
VGG16_NOTOP_URL = ("https://github.com/fchollet/deep-learning-models/releases/download/"
                   "v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
