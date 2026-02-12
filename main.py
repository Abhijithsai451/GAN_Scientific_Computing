from data_processing.dataloader import get_dataloaders
from evaluation.visualize import visualize_batch
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



if __name__ == "__main__":
    main()