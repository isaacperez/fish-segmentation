import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt
import numpy as np
import cv2 

import torch

import dataset
import CONST
import data_augmentation

# NOTE: ia.seed(randomInteger) can be use to perform the same augmentation 
# technique twice with the same seed


# -----------------------------------------------------------------------------
# Load the images
# -----------------------------------------------------------------------------
current_set = dataset.CustomImageDataset(CONST.VAL_FOLDER_NAME, do_data_augmentation=False)
current_dataset = torch.utils.data.DataLoader(current_set, batch_size=1, shuffle=True, num_workers=2)

for idx, (image, mask) in enumerate(current_dataset):

    # Transform to numpy 
    image = (image * 255).numpy().astype(np.uint8).squeeze(0).transpose((1, 2, 0))
    mask = mask.numpy().squeeze(0).astype(np.uint8)

    # -----------------------------------------------------------------------------
    # Apply random transformation
    # -----------------------------------------------------------------------------
    image_aug, mask_aug = data_augmentation.seq(image=image, segmentation_maps=SegmentationMapsOnImage(mask, shape=image.shape))
    mask_aug = mask_aug.get_arr()

    # -----------------------------------------------------------------------------
    # Show the results
    # -----------------------------------------------------------------------------
    plt.figure(1)
    plt.title("Original image")
    plt.imshow(image)
    plt.draw()

    print("\n", "-"*80, sep="")
    print("Current idx:", idx)
    print("-"*80)

    print("Original image info:")
    print("\tMean:".ljust(8), np.round(np.mean(image, axis=(0, 1)), 2))
    print("\tMin:".ljust(8), np.round(np.min(image, axis=(0, 1)), 2))
    print("\tMax:".ljust(8), np.round(np.max(image, axis=(0, 1)), 2))

    plt.figure(2)
    plt.title("Augmented image")
    plt.imshow(image_aug)
    plt.draw()

    print("\nAugmented image info:")
    print("\tMean:".ljust(8), np.round(np.mean(image_aug, axis=(0, 1)), 2))
    print("\tMin:".ljust(8), np.round(np.min(image_aug, axis=(0, 1)), 2))
    print("\tMax:".ljust(8), np.round(np.max(image_aug, axis=(0, 1)), 2))

    plt.figure(3)
    plt.title("Original mask")
    plt.imshow(mask)
    plt.draw()

    print("Original mask info:")
    print("\tMean:".ljust(8), np.round(np.mean(mask, axis=(0, 1)), 2))
    print("\tMin:".ljust(8), np.round(np.min(mask, axis=(0, 1)), 2))
    print("\tMax:".ljust(8), np.round(np.max(mask, axis=(0, 1)), 2))

    plt.figure(4)
    plt.title("Augmented mask")
    plt.imshow(mask_aug)
    plt.draw()

    print("\nAugmented mask info:")
    print("\tMean:".ljust(8), np.round(np.mean(mask_aug, axis=(0, 1)), 2))
    print("\tMin:".ljust(8), np.round(np.min(mask_aug, axis=(0, 1)), 2))
    print("\tMax:".ljust(8), np.round(np.max(mask_aug, axis=(0, 1)), 2))

    plt.show(block=False)

    # Wait break only with a keyboard press
    while not plt.waitforbuttonpress():
        pass
