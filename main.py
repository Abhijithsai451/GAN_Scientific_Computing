from data_processing.dataloader import get_dataloaders
from evaluation.visualize import visualize_batch
from models.discriminator import Discriminator
from models.generator import Generator
from models.model_utils import weight_initialization
from utils.config_parser import get_args, Config
from utils.logger_config import setup_logger, tensorboard_logger
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory")

def main():
    # Loading the Configuration file passed in Runtime.
    config_path = get_args()
    config = Config(config_path)

    # Setup Logging
    logger = setup_logger(config)
    writer = tensorboard_logger(config)

    logger.info(f"Starting the Project: {config.project_name} ")
    logger.info(f"Configuration Loaded from {config_path} ")

    train_loader, test_loader, valid_loader = get_dataloaders(config)
    logger.info(f"Imported the dataset from {config.dataset.get("name")} ")

    # Visualizing the Imported Dataset
    visualize_batch(
                    train_loader,
                    class_names=config.dataset.get("classes"),
                    num_images=16,

    )
    # Building Model Architectures
    logger.info(f"Building the {config.project_name} Architecture from the config file  ")
    generator = Generator(config).to(config.device)
    logger.info("Generator built successfully")
    discriminator = Discriminator(config).to(config.device)
    logger.info("Discriminator built successfully")

    # Initializing the weights
    generator.apply(weight_initialization)
    discriminator.apply(weight_initialization)
    logger.info("Weights initialized on Generator and Discriminator")





    logger.info(f"Finished the Project: {config.project_name} ")
if __name__ == "__main__":
    main()