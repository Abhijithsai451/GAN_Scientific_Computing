import os
import torch
from torchvision.utils import save_image
from data_processing.dataloader import get_dataloaders
from evaluation.visualize import visualize_batch
from models.discriminator import Discriminator
from models.generator import Generator
from models.model_utils import weight_initialization
from training.train import GANTrainer
from utils.config_parser import get_args, Config
from utils.logger_config import setup_logger
import warnings

from utils.tensorboard_logger import TensorBoardLogger

warnings.filterwarnings("ignore", message=".*pin_memory")

def main():
    # Loading the Configuration file passed in Runtime.
    config_path = get_args()
    config = Config(config_path)

    # Setup Logging
    logger = setup_logger(config)

    DEVICE = config.device if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Starting the Project: {config.project_name} ")
    logger.info(f"Configuration Loaded from {config_path} ")

    train_loader, test_loader, valid_loader = get_dataloaders(config)
    logger.info(f"Imported the dataset from {config.dataset.get("name")} ")

    # Visualizing the Imported Dataset
    """visualize_batch(
                    train_loader,
                    class_names=config.dataset.get("classes"),
                    num_images=16,

    )"""
    # Building Model Architectures
    logger.info(f"Building the {config.project_name} Architecture from the config file  ")
    generator = Generator(config).to(DEVICE)
    logger.info("Generator built successfully")
    discriminator = Discriminator(config).to(DEVICE)
    logger.info("Discriminator built successfully")

    # Initializing the weights
    generator.apply(weight_initialization)
    discriminator.apply(weight_initialization)
    logger.info("Weights initialized on Generator and Discriminator")

    # Initialize the Trainer & Tensorboard Logger
    trainer = GANTrainer(generator, discriminator, config,DEVICE)
    tb_logger  = TensorBoardLogger(config)
    os.makedirs("results/samples", exist_ok=True)

    # Creating Noise for the Generator
    fixed_noise = torch.randn(64, config.model['latent_dim']).to(DEVICE)
    fixed_labels = torch.arange(0,10).repeat(7)[:64].to(DEVICE)

    # Training Loop
    logger.info(f"Model Training Initialized......")
    for epoch in range(config.trainer['epochs']):
        temp_loss_g = 0.0
        temp_loss_d = 0.0
        total_d_x = 0.0
        total_d_gz = 0.0

        for i, (real_imgs, labels) in enumerate(train_loader):
            real_imgs, labels = real_imgs.to(DEVICE), labels.to(DEVICE)
            loss_d, loss_g, d_x, d_gz = trainer.train_step(real_imgs, labels)
            temp_loss_g += loss_g
            temp_loss_d += loss_d
            total_d_x += d_x
            total_d_gz += d_gz
            if i % 100 == 0:
                print(f"Epoch [{epoch}] Batch [{i}/{len(train_loader)}] | Loss D: {loss_d:.4f} | Loss G: {loss_g:.4f} ")


        # Metrics Logging
        num_batches = len(train_loader)
        logger.info(f"Number of Batches [{num_batches}]")
        avg_loss_g = temp_loss_g / num_batches
        avg_loss_d = temp_loss_d / num_batches
        avg_d_x = total_d_x / num_batches
        avg_d_gz = total_d_gz / num_batches
        print(f"==> Epoch [{epoch}] Summary | Loss D: {avg_loss_d:.4f} | Loss G: {avg_loss_g:.4f} | D(x): {avg_d_x:.4f} | G(x): {avg_d_gz:.4f} ")
        tb_logger.log_epoch(epoch, avg_loss_d, avg_loss_g, avg_d_x, avg_d_gz)
        # TODO

    logger.info(f"Finished the Project: {config.project_name} ")
if __name__ == "__main__":
    main()