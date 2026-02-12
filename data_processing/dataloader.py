import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from utils.logger_config import get_logger
from data_processing.data_transformer import get_train_transforms, get_test_transforms
from sklearn.model_selection import train_test_split

logger = get_logger()
def get_dataloaders(config):
    """
    Creates production-grade DataLoaders for CIFAR-10.
    Ensures matching ranges for Generator Tanh output [-1, 1].
    """
    # 1. Define Normalization (transforms [0,1] to [-1,1])
    # Formula: normalized = (x - mean) / std
    logger.info("Loading CIFAR-10 dataset...")
    tin_mean = config.data_transform.get("tin_mean")
    tin_std = config.data_transform.get("tin_std")
    num_aug_images = config.data_transform.get("aug_images")
    use_pin = config.use_pin

    # Initialize Datasets
    train_dataset = datasets.CIFAR10(
        root=config.data['root'],
        train=True,
        download=config.data['download'],
        transform= get_train_transforms(tin_mean, tin_std)
    )

    full_test_dataset = datasets.CIFAR10(
        root=config.data['root'],
        train=False,
        download=config.data['download'],
    )

    indices = list(range(len(full_test_dataset)))
    valid_indices, test_indices = train_test_split(
        indices, test_size=0.5, random_state=42, stratify=full_test_dataset.targets
    )
    test_dataset = Subset(full_test_dataset, test_indices)
    valid_dataset = Subset(full_test_dataset, valid_indices)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer['batch_size'],
        shuffle=True,
        num_workers=config.data.get('num_workers', 2),
        pin_memory=use_pin
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.trainer['batch_size'],
        shuffle=False,
        num_workers=config.data.get('num_workers', 2),
        pin_memory=use_pin
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.trainer['batch_size'],
        shuffle=False,
        num_workers=config.data.get('num_workers', 2))

    logger.info(f"Dataloaders ready. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, test_loader , valid_loader
