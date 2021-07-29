import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np 
import torch 

import CONST


# -----------------------------------------------------------------------------
# imgaug probailities
# -----------------------------------------------------------------------------
rotation_prob = 0.3
affine_prob = 0.3
flip_lr_prob = 0.5
flip_ud_prob = 0.0

arithmetic_prob = 0.3
arithmetic_same_time_ops = 2

cutout_prob = 0.3
quality_prob = 0.4

jigsaw_prob = 0.3

# -----------------------------------------------------------------------------
# Define the transformations
# -----------------------------------------------------------------------------
seq = iaa.Sequential([
        iaa.Sometimes(rotation_prob, iaa.Affine(rotate=(-360, 360))),
        iaa.Sometimes(affine_prob, iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, 
                                      translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                      shear=(-16, 16))),
        iaa.Fliplr(flip_lr_prob),
        iaa.Flipud(flip_ud_prob),
        iaa.Sometimes(arithmetic_prob, iaa.SomeOf(arithmetic_same_time_ops, [
            iaa.Add((-20, 20), per_channel=0.5),
            iaa.Multiply((0.7, 1.3), per_channel=0.5),
            iaa.AdditiveGaussianNoise(scale=3)], random_order=True)),

        iaa.Sometimes(cutout_prob, iaa.SomeOf(1, [
            iaa.CoarseDropout(0.02, size_percent=0.5),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            iaa.SaltAndPepper(0.05),
            iaa.SaltAndPepper(0.05, per_channel=True),
            iaa.CoarseSaltAndPepper(0.02, size_percent=(0.01, 0.05), 
                per_channel=True)])),
        iaa.Sometimes(quality_prob, iaa.SomeOf(1, [
            iaa.JpegCompression(compression=(70, 99)),
            iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0)),
            iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(5)),
            iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(20)),
            iaa.GaussianBlur(sigma=(0.0, 1.5)),
            iaa.MedianBlur(k=(3, 5)),
            iaa.MotionBlur(k=7),
            iaa.WithHueAndSaturation(
                iaa.WithChannels(0, iaa.Add((0, 20)))),
            iaa.MultiplyHue((0.85, 1.25)),
            iaa.MultiplySaturation((0.75, 1.35)),
            iaa.ChangeColorTemperature((3000, 10000)),
            iaa.GammaContrast((0.8, 1.2)),
            iaa.AllChannelsHistogramEqualization(),
            iaa.Sharpen(alpha=(0.05, 0.2), lightness=(0.75, 1.0)),
            iaa.Emboss(alpha=(0.3, 0.6), strength=(0.5, 1.5)),
            iaa.AveragePooling(2),
            iaa.MedianPooling(2)
            ])),

        iaa.Sometimes(jigsaw_prob, iaa.Jigsaw(nb_rows=(1, 2), nb_cols=(1, 2), allow_pad=False)),
        #iaa.CropToFixedSize(width=CONST.WIDTH_TRAIN, height=CONST.HEIGHT_TRAIN, position="uniform")       
    ]) 


def dataAugmentation(image, mask):
    """This function receives an image and its masks and apply random data augmentation techniques to both the image and 
       the mask"""
    # Transform to numpy 
    image = image.numpy().transpose((1, 2, 0))
    mask = mask.numpy()

    # -----------------------------------------------------------------------------
    # Apply random transformation
    # -----------------------------------------------------------------------------
    image_aug, mask_aug = seq(image=image, segmentation_maps=SegmentationMapsOnImage(mask, shape=image.shape))
    mask_aug = mask_aug.get_arr()

    # Transform back to pytorch
    image_aug = torch.from_numpy(image_aug.transpose((2, 0, 1)))
    mask_aug = torch.from_numpy(mask_aug)

    return image_aug, mask_aug