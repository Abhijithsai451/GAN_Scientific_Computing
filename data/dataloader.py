from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.logger_config import get_logger
from data.data_transformer import get_train_transforms, get_test_transforms
from  utils.randomizer_config import set_seed

logger = get_logger()
def get_dataloaders(config):
    """
    Creates production-grade DataLoaders for CIFAR-10.
    Ensures matching ranges for Generator Tanh output [-1, 1].
    """
    # 1. Define Normalization (transforms [0,1] to [-1,1])
    # Formula: normalized = (x - mean) / std
    stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # 2. Training Transforms: Includes Augmentation for Discriminator
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
    ])

    # 3. Test Transforms: Only ToTensor and Normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    logger.info("Loading CIFAR-10 dataset...")

    # 4. Initialize Datasets
    train_dataset = datasets.CIFAR10(
        root=config.data['root'],
        train=True,
        download=config.data['download'],
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=config.data['root'],
        train=False,
        download=config.data['download'],
        transform=test_transform
    )

    # 5. Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer['batch_size'],
        shuffle=True,
        num_workers=config.data.get('num_workers', 2),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.trainer['batch_size'],
        shuffle=False,
        num_workers=config.data.get('num_workers', 2)
    )

    logger.info(f"Dataloaders ready. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, test_loader
"""
def get_dataloaders(config):

    train_dataset = datasets.CIFAR10(
                        root=root,
                        train= True,
                        download=True,
                        transform=get_train_transforms()
                        )

    test_dataset = datasets.CIFAR10(
                        root = root,
                        train=False,
                        download=True,
                        transform=get_test_transforms()
                        )

    return train_dataset , test_dataset

#aug_dataset = augment_data(train_dataset, aug_dir,num_aug_images,aug_exist)
"""