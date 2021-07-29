import os

import torch 


#-----------------------------------------------------------------------------------------------------------------------
# Dataset constants
#-----------------------------------------------------------------------------------------------------------------------
DATASET_PARENT_FOLDER = "../datasets/DEEPFISH_AND_LUDERICK_DATASET/"


TRAIN_FOLDER_NAME = "TRAIN"
VAL_FOLDER_NAME = "VAL"
TEST_FOLDER_NAME = "TEST"

IMAGES_FOLDER_NAME = "images"
MASKS_FOLDER_NAME = "masks"

IMAGES_FILE_EXT = ".jpg"
MASKS_FILE_EXT = ".png"

HEIGHT = 1072 // 2
WIDTH = 1920 // 2

HEIGHT_TRAIN = 512
WIDTH_TRAIN = 512

CLASSES = ["BACKGROUND", "FOREGROUND"]

WEIGHTS = {
    "DEEPFISH_DATASET": torch.Tensor([1, 49]),
    "ICFUSH_DATASET": torch.Tensor([1, 49]),
    "DEEPFISH_AND_LUDERICK_DATASET": torch.Tensor([1, 49])
    }

IMG_EXTENSION = [".png", ".PNG", ".jpg", ".JPG"]

# Checks
assert DATASET_PARENT_FOLDER.endswith(os.sep)

assert os.sep not in TRAIN_FOLDER_NAME
assert os.sep not in VAL_FOLDER_NAME
assert os.sep not in TEST_FOLDER_NAME

assert os.sep not in IMAGES_FOLDER_NAME
assert os.sep not in MASKS_FILE_EXT

assert IMAGES_FILE_EXT.startswith(".")
assert MASKS_FILE_EXT.startswith(".")

# Metrics
MEAN_LOSS_TRAIN = "MEAN_LOSS_TRAIN"
MEAN_LOSS_VAL = "MEAN_LOSS_VAL"
MEAN_ACCURACY = "MEAN_ACCURACY"
MEAN_RECALL = "MEAN_RECALL"
MEAN_F1_SCORE = "MEAN_F1_SCORE"
MEAN_PRECISION = "MEAN_PRECISION"
MEAN_IOU = "MEAN_IOU"

ACCURACY = "ACCURACY"
RECALL = "RECALL"
F1_SCORE = "F1_SCORE"
PRECISION = "PRECISION"
IOU = "IOU"