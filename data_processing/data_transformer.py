import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from utils.randomizer_config import set_seed

def get_train_transforms(tin_mean, tin_std):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean = tin_mean, std = tin_std)
    ])
    return transform

def get_test_transforms(tin_mean, tin_std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = tin_mean, std = tin_std)
    ])
    return transform

def get_aug_transforms(tin_mean, tin_std):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean = tin_mean, std = tin_std)
    ])
    return transform
