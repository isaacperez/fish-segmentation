import os

import torch
import torch.nn.functional as F
import torch.nn as nn

import dataset 
import CONST


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        #self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(CONST.WEIGHTS[CONST.DATASET_PARENT_FOLDER.split(os.sep)[-2]]))
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def smooth_one_hot(self, targets):
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method

        """
        targets = torch.where(targets==0, 0.1, 0.9)
        return targets

    def forward(self, inputs, targets):
        
        # Get the loss
        loss = self.loss(inputs, targets)
        
        # Extract the loss for the foreground and background classes
        background_loss = torch.masked_select(loss, targets == 0)
        foreground_loss = torch.masked_select(loss, targets == 1)

        # Select the same amount of loss values for the background class
        if foreground_loss.numel() <= background_loss.numel() and foreground_loss.numel() > 0:
            index_for_background = torch.randperm(background_loss.numel())[:foreground_loss.numel()]
            background_loss = background_loss[index_for_background]


        # Concat both losses and extract the mean value
        loss = torch.cat([background_loss, foreground_loss], dim=0)
    
        loss = torch.mean(loss)

        return loss

        """
        # DICE LOSS
        # compute softmax over the classes axis
        input_soft = F.softmax(inputs, dim=1)
        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(input_soft[:, 1, :, :] * targets, dims)
        cardinality = torch.sum(input_soft[:, 1, :, :] + targets, dims)

        dice_score = 2. * intersection / (cardinality + 1e-06)

        return torch.mean(1. - dice_score)
        """
        """
        smooth_label = self.smooth_one_hot(targets)
        
        m = nn.LogSoftmax(dim=1)
        softmax = m(inputs)
        
        softmax_background = softmax[:, 0, :, :]
        softmax_foreground = softmax[:, 1, :, :]

        cross_entropy_background = (1-targets) * -softmax_background * smooth_label
        cross_entropy_foreground = targets * -softmax_foreground * smooth_label * 49

        cross_entropy_loss = cross_entropy_background + cross_entropy_foreground

        loss = torch.mean(cross_entropy_loss)

        return loss
        """
        """
        m = nn.LogSoftmax(dim=1)
        softmax = m(inputs)
        
        softmax_background = softmax[:, 0, :, :]
        softmax_foreground = softmax[:, 1, :, :]

        cross_entropy_background = (1-targets) * -softmax_background
        cross_entropy_foreground = targets * -softmax_foreground * 49   #TODO SE TIENE EN CUENTA TODA LA IMAGEN

        cross_entropy_loss = cross_entropy_background + cross_entropy_foreground

        loss = torch.mean(cross_entropy_loss)

        return loss
        """

if __name__ == '__main__':

    current_set = dataset.CustomImageDataset(CONST.VAL_FOLDER_NAME)
    current_dataset = torch.utils.data.DataLoader(current_set, batch_size=1, shuffle=False, num_workers=2)

    loss = Loss()

    for idx, (img, mask) in enumerate(current_dataset):
        
        if idx < 1:
            continue
        
        print(torch.count_nonzero(mask) / (mask.shape[1] * mask.shape[2]))
        print(loss(torch.ones_like(F.one_hot(mask, num_classes=2).permute([0, 3, 1, 2]).type(torch.float32)), mask))
        print(loss(F.one_hot(mask, num_classes=2).permute([0, 3, 1, 2]).type(torch.float32) * 100, mask))
        print(loss(torch.rand(F.one_hot(mask, num_classes=2).permute([0, 3, 1, 2]).type(torch.float32).size()), mask))
        exit()