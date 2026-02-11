from data import dataloader
from data.dataloader import get_dataloaders
from models.discriminator import Discriminator
from models.generator import Generator
from utils.config_parser import get_args, Config
from utils.logger_config import setup_logger, tensorboard_logger
from utils.randomizer_config import set_seed


def main():
    # Loading the Configuration file passed in Runtime.
    config_path = get_args()
    config = Config(config_path)

    # Setup Logging
    logger = setup_logger(config)
    writer = tensorboard_logger(config)

    logger.info(f"Starting the Project: {config.project_name} ")
    logger.info(f"Configuration Loaded from {config_path} ")

    train_loader, test_loader = get_dataloaders(config)

    # Debug: Check a single batch
    images, labels = next(iter(train_loader))
    logger.info(f"Batch Image Shape: {images.shape}")  # Should be [Batch, 3, 32, 32]
    logger.info(f"Value Range: {images.min().item():.2f} to {images.max().item():.2f}")  # Should be ~ -1 to 1


if __name__ == "__main__":
    main()