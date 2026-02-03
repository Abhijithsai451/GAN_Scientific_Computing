import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from utils.random_config import set_seed

set_seed(42)
TIN_MEAN = (0.5, 0.5, 0.5)
TIN_STD = (0.5, 0.5, 0.5)
num_aug_images = 10
# To run the augmenting in a device with GPU.
#device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")

# To run the augmenting on GPU in Apple Silicon Device.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Comment the below line if cuda is present.
#device = torch.device("cpu")


def get_train_transforms():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean = TIN_MEAN, std = TIN_STD)
    ])
    return transform

def get_test_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TIN_MEAN, std=TIN_STD)
    ])
    return transform


def get_aug_transforms():
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=TIN_MEAN, std=TIN_STD)
    ])
    return transform
