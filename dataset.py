from torchvision.transforms.functional import InterpolationMode
from CONST import IMAGES_FOLDER_NAME
import os
import glob 

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import CenterCrop
from torchvision.transforms import Resize

import CONST
import data_augmentation


crop = CenterCrop((CONST.HEIGHT * 2, CONST.WIDTH * 2))
resizeImage = Resize((CONST.HEIGHT, CONST.WIDTH))
resizeMask = Resize((CONST.HEIGHT, CONST.WIDTH), interpolation=InterpolationMode.NEAREST)
def crop_image(image, mask):
    image = crop(image)
    mask = crop(mask)
    
    image = resizeImage(image)
    mask = resizeMask(mask)
    
    # Get non-zero indexes
    (C_index, Y_index, X_index) = torch.nonzero(mask, as_tuple=True)
    
    if X_index.numel() == 0:
        start_X = torch.randint(0, CONST.WIDTH - CONST.WIDTH_TRAIN - 2, (1,))
        start_Y = torch.randint(0, CONST.HEIGHT - CONST.HEIGHT_TRAIN - 2, (1,))

        final_X = start_X + CONST.WIDTH_TRAIN
        final_Y = start_Y + CONST.HEIGHT_TRAIN
        
    else:
        X_index = X_index.to(torch.float32)
        Y_index = Y_index.to(torch.float32)

        mean_X = X_index.mean()
        mean_Y = Y_index.mean()

        # A random displacement
        mean_X = torch.clip(mean_X + torch.randint(- int(CONST.WIDTH_TRAIN / 4), int(CONST.WIDTH_TRAIN / 4), (1,)), 0, CONST.WIDTH)
        mean_Y = torch.clip(mean_Y + torch.randint(- int(CONST.HEIGHT_TRAIN / 4), int(CONST.HEIGHT_TRAIN / 4), (1,)), 0, CONST.HEIGHT)

        # Define the final position
        start_X = (mean_X - CONST.WIDTH_TRAIN / 2).to(torch.int32)
        if start_X < 0:
            start_X = 0
        start_Y = (mean_Y - CONST.HEIGHT_TRAIN / 2).to(torch.int32)
        if start_Y < 0:
            start_Y = 0

        # Check the dispersion
        dispersion_X = X_index.max() - X_index.min()
        dispersion_Y = Y_index.max() - Y_index.min()

        if dispersion_X > CONST.WIDTH_TRAIN or dispersion_Y > CONST.HEIGHT_TRAIN:

            # Move the window to the biggest area
            if X_index.max() - mean_X < mean_X - X_index.min():

                # Move to the max
                mean_X = X_index.max()
                mean_Y = Y_index.max()
        
                # A random displacement
                mean_X = torch.clip(mean_X + torch.randint(- int(CONST.WIDTH_TRAIN / 4), int(CONST.WIDTH_TRAIN / 4), (1,)), 0, CONST.WIDTH)
                mean_Y = torch.clip(mean_Y + torch.randint(- int(CONST.HEIGHT_TRAIN / 4), int(CONST.HEIGHT_TRAIN / 4), (1,)), 0, CONST.HEIGHT)

                # Define the final position
                start_X = (mean_X - CONST.WIDTH_TRAIN).to(torch.int32)
                if start_X < 0:
                    start_X = 0
                start_Y = (mean_Y - CONST.HEIGHT_TRAIN).to(torch.int32)
                if start_Y < 0:
                    start_Y = 0

            else:

                # Move to the min
                mean_X = X_index.min()
                mean_Y = Y_index.min()

                # A random displacement
                mean_X = torch.clip(mean_X + torch.randint(- int(CONST.WIDTH_TRAIN / 4), int(CONST.WIDTH_TRAIN / 4), (1,)), 0, CONST.WIDTH)
                mean_Y = torch.clip(mean_Y + torch.randint(- int(CONST.HEIGHT_TRAIN / 4), int(CONST.HEIGHT_TRAIN / 4), (1,)), 0, CONST.HEIGHT)

                # Define the final position
                start_X = (mean_X).to(torch.int32)
                start_Y = (mean_Y).to(torch.int32)
        
        final_X = start_X + CONST.WIDTH_TRAIN
        if final_X >= CONST.WIDTH:
            final_X = CONST.WIDTH - 1
            start_X = final_X - CONST.WIDTH_TRAIN
        
        final_Y = start_Y + CONST.HEIGHT_TRAIN
        if final_Y >= CONST.HEIGHT:
            final_Y = CONST.HEIGHT - 1
            start_Y = final_Y - CONST.HEIGHT_TRAIN

    assert start_X >= 0 and start_X < final_X
    assert start_Y >= 0 and start_Y < final_Y
    assert final_X < CONST.WIDTH and final_Y < CONST.HEIGHT
    assert final_X - start_X == CONST.WIDTH_TRAIN
    assert final_Y - start_Y == CONST.HEIGHT_TRAIN

    # Extract the region of interest
    image = image[:, start_Y:final_Y, start_X:final_X]
    mask = mask[:, start_Y:final_Y, start_X:final_X]

    return image, mask


class CustomImageDataset(Dataset):
    def __init__(self, dataset_folder_name, do_data_augmentation):
        self.images_paths = glob.glob(CONST.DATASET_PARENT_FOLDER + dataset_folder_name + os.sep
                                      + CONST.IMAGES_FOLDER_NAME + os.sep + "*" + CONST.IMAGES_FILE_EXT)
        self.masks_paths = glob.glob(CONST.DATASET_PARENT_FOLDER + dataset_folder_name + os.sep
                                     + CONST.MASKS_FOLDER_NAME + os.sep + "*" + CONST.MASKS_FILE_EXT)

        # Sort the paths so they can match
        self.images_paths.sort()
        self.masks_paths.sort()

        self.do_data_augmentation = do_data_augmentation

        # Checks
        assert len(self.images_paths) == len(self.masks_paths)
        assert len(self.images_paths) > 0
        assert all([img_name.split(os.sep)[-1].split(".")[0] == mask_name.split(os.sep)[-1].split(".")[0] 
                    for img_name, mask_name in zip(self.images_paths, self.masks_paths)])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):

        # Read the image and the mask
        image = read_image(self.images_paths[idx])
        mask = read_image(self.masks_paths[idx])

        # Remove the alpha channel from the image if it has one
        if image.shape[0] == 4:
            image = image[:3, :, :]

        # Give the image the desire size
        image, mask = crop_image(image, mask)

        # Remove the channel dimension of the mask
        mask = mask[0, :, :]

        # Do data augmentation
        if self.do_data_augmentation: 
            image, mask = data_augmentation.dataAugmentation(image, mask)

        # Transform the data to float and put in the range [0, 1]
        image = image.type(torch.float32) / 255.0
        mask = (mask / 255).type(torch.int64)

        return image, mask


# ----------------------------------------------------------------------------------------------------------------------
# Testing code
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    current_set = CustomImageDataset(CONST.TRAIN_FOLDER_NAME, do_data_augmentation=True)
    dataset = torch.utils.data.DataLoader(current_set, batch_size=1, shuffle=False, num_workers=1)

    print(len(dataset))

    for idx, (img, mask) in enumerate(dataset):
        print(idx, img.shape, mask.shape)
        # Remove batch dimension
        #img = img[0, :, :, :]
        #mask = mask[0, :, :]
        
        # In Pytorch, the first dimension is used for channels
        #img = img.permute(1, 2, 0)
        """
        print("Displaying data ", idx, end="\r")

        plt.figure(1)
        plt.imshow((img * 255).type(torch.uint8))

        plt.figure(2)
        plt.imshow((mask * 255).type(torch.uint8), cmap="gray")

        plt.show()
        """

        assert all([list(img.shape) == [1, 3, CONST.HEIGHT_TRAIN, CONST.WIDTH_TRAIN]])
        assert all([list(mask.shape) == [1, CONST.HEIGHT_TRAIN, CONST.WIDTH_TRAIN]])
        