import os
import torch
from torchvision.utils import save_image
from data_processing.dataloader import get_dataloaders
from evaluation.evaluate import generate_plots
from models.discriminator import Discriminator
from models.generator import Generator
import wandb as wb
from models.model_utils import weight_initialization
from training.train import GANTrainer
from utils.config_parser import get_args
from utils.logger_config import setup_logger
from wandb_utils.wandb_config import WandBConfig


def main():
    # Importing the config file from the command line
    config_path = get_args()

    # Setting up the WandB Logger
    wb_logger = WandBConfig(config_path)
    config =  wb_logger.get_config()

    logger = setup_logger(config)
    DEVICE = config.device if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Starting the Project: {config.project_name} on {DEVICE} ")
    logger.info(f"Configuration Loaded from {config_path} ")

    train_loader, test_loader, valid_loader = get_dataloaders(config)
    logger.info(f"Imported the dataset from {config.dataset.get("name")} ")

    real_batch, _ = next(iter(train_loader))
    wb_logger.log_images(real_batch, "Batch of Real Images")

    # Building Model Architectures
    logger.info(f"Building the {config.project_name} Architecture from the config file  ")
    generator = Generator(config).to(DEVICE)
    logger.info("Generator built successfully")
    discriminator = Discriminator(config).to(DEVICE)
    logger.info("Discriminator built successfully")

    wb.watch(generator, log="all", log_freq=5)
    wb.watch(discriminator, log="all", log_freq=5)

    logger.info("Models are now being 'watched' by WandB")

    # Initializing the weights
    generator.apply(weight_initialization)
    discriminator.apply(weight_initialization)
    logger.info("Weights initialized on Generator and Discriminator")

    # Initialize the Trainer & Tensorboard Logger
    trainer = GANTrainer(generator, discriminator, config, DEVICE)
    os.makedirs("results/samples", exist_ok=True)
    os.makedirs(config.logger['ckpt_dir'], exist_ok=True)

    # Creating Noise for the Generator
    fixed_noise = torch.randn(64, config.model['latent_dim']).to(DEVICE)
    fixed_labels = torch.arange(0, 10).repeat(7)[:64].to(DEVICE)

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
        wb_logger.log_step(metrics = {"loss_d": avg_loss_d, "loss_g": avg_loss_g, "d_x": avg_d_x, "d_gz": avg_d_gz}, step=epoch)
        with (torch.no_grad()):
            fake_samples = generator(fixed_noise, fixed_labels)
            wb_logger
            save_image(fake_samples, f"results/samples/epoch_{epoch}_{config.project_name}.png", normalize=True)
            wb_logger.log_images(fake_samples, f"Epoch {epoch} Samples")
    logger.info("Saving the generator's and disciminator's Checkpoints and state_dict")
    torch.save(generator.state_dict(), os.path.join(config.logger['ckpt_dir'], "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(config.logger['ckpt_dir'], "discriminator_final.pth"))

    logger.info("Plotting the Loss Graphs")
    generate_plots("results", "results/improved")
    logger.info(f"Finished the Project: {config.project_name} ")

    artifact = wb.Artifact(f"{config.project_name}_model", type="model")
    artifact.add_file(os.path.join(config.logger['ckpt_dir'], "generator_final.pth"))
    artifact.add_file(os.path.join(config.logger['ckpt_dir'], "discriminator_final.pth"))
    wb.log_artifact(artifact)
    logger.info("Model Artifacts logged to WandB Artifact Registry")

    wb.finish()

if __name__ == '__main__':
    main()