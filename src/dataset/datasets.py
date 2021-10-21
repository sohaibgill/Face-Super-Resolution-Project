import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):

    def __init__(self, hr_root, lr_root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform_hr = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std),
             ]
        )

        self.hr_files = sorted(glob.glob(hr_root + "/*.*"))
        self.lr_files = sorted(glob.glob(lr_root + "/*.*"))

    def __getitem__(self, index):
        img_hr = Image.open(self.hr_files[index])
        img_lr = Image.open(self.lr_files[index])

        img_lr = self.transform(img_lr)
        img_hr = self.transform_hr(img_hr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.hr_files)
