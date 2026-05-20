import os
import torch
from torchvision.utils import save_image
from data_processing.dataloader import get_dataloaders
from evaluation.evaluate import generate_plots
from evaluation.visualize import visualize_batch
from models.discriminator import Discriminator
from models.generator import Generator
from models.model_utils import weight_initialization
from training.train import GANTrainer
from utils.config_parser import get_args, Config
from utils.logger_config import setup_logger
import warnings
import wandb
from wandb_utils.wandb_config import WandBConfig
from utils.tensorboard_logger import TensorBoardLogger

warnings.filterwarnings("ignore", message=".*pin_memory")

def main():
    # Loading the Configuration file passed in Runtime.
    
    config_path = get_args()
    config = Config(config_path)

    # Setup Logging
    logger = setup_logger(config)
    wb_logger = WandBConfig(config_path , job_type="train")
    wandb_config = wb_logger.get_config()

    DEVICE = config.device if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Starting the Project: {config.project_name} on {DEVICE} ")
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

    # Log models to wandb to track gradients and parameters.
    wandb.watch(generator, log="all", log_freq=config.logger.get('save_every', 3))
    wandb.watch(discriminator, log="all", log_freq=config.logger.get('save_every', 3))
    logger.info("Models are now being 'watched' by WandB")

    # Initializing the weights
    generator.apply(weight_initialization)
    discriminator.apply(weight_initialization)
    logger.info("Weights initialized on Generator and Discriminator")


    trainer = GANTrainer(generator, discriminator, config,DEVICE)
    os.makedirs("results/samples", exist_ok=True)
    os.makedirs(config.logger['ckpt_dir'], exist_ok=True)

    # Creating Noise for the Generator
    fixed_noise = torch.randn(64, config.model['latent_dim']).to(DEVICE)
    fixed_labels = torch.arange(0,10).repeat(7)[:64].to(DEVICE)

    # Log a batch of real images
    real_batch, _ = next(iter(train_loader))
    wb_logger.log_images(real_batch, "Initial Real Images (Overview)")
    logger.info("Initial real image batch logged to WandB overview.")

    real_images_table = wandb.Table(columns=["image", "label"])
    for idx, img_tensor in enumerate(real_batch):
        if idx < 64:
            real_images_table.add_data(wandb.Image(img_tensor), idx % 10)
        else:
            break
    wb_logger.log_step({"Initial Real Images Gallery": real_images_table})
    logger.info("Initial real image batch logged to WandB gallery")

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

        wb_logger.log_step(metrics = {
            "loss_d": avg_loss_d,
            "loss_g": avg_loss_g,
            "d_x": avg_d_x,
            "d_gz": avg_d_gz
        }, step=epoch)

        with torch.no_grad():
            fake_samples = generator(fixed_noise, fixed_labels)
            save_image(fake_samples, f"results/samples/epoch_{epoch}_{config.project_name}.png", normalize=True)
            wb_logger.log_images(fake_samples, f"Generated Samples (Epoch {epoch})", step=epoch)
            logger.info(f"Generated samples for epoch {epoch} logged to WandB overview.")

            generated_images_table = wandb.Table(columns=["generated_image", "epoch", "generated_label"])
            for idx, img_tensor in enumerate(fake_samples):
                generated_images_table.add_data(wandb.Image(img_tensor), epoch, fixed_labels[idx].item())
            wb_logger.log_step({f"Generated Samples Gallery": generated_images_table}, step=epoch)
            logger.info(f"Generated samples for epoch {epoch} logged to WandB gallery.")

        if (epoch + 1) % config.logger.get('save_every', 10) == 0:
            ckpt_gen_path = os.path.join(config.logger['ckpt_dir'], f"generator_epoch_{epoch+1}.pth")
            ckpt_disc_path = os.path.join(config.logger['ckpt_dir'], f"discriminator_epoch_{epoch+1}.pth")
            torch.save(generator.state_dict(), ckpt_gen_path)
            torch.save(discriminator.state_dict(), ckpt_disc_path)

            ckpt_artifact = wandb.Artifact(f"{config.project_name}_checkpoint_epoch_{epoch+1}", type="model")
            ckpt_artifact.add_file(ckpt_gen_path)
            ckpt_artifact.add_file(ckpt_disc_path)
            wb_logger.run.log_artifact(ckpt_artifact)
            logger.info(f"Model checkpoint for epoch {epoch+1} logged as WandB artifact.")

    logger.info("Saving the generator's and disciminator's Checkpoints and state_dict")
    gen_final_path = os.path.join(config.logger['ckpt_dir'], "generator_final.pth")
    disc_final_path = os.path.join(config.logger['ckpt_dir'], "discriminator_final.pth")
    torch.save(generator.state_dict(), gen_final_path)
    torch.save(discriminator.state_dict(), disc_final_path)

    # Log model checkpoints as W&B Artifacts
    artifact = wandb.Artifact(f"{config.project_name}_model", type="model")
    artifact.add_file(gen_final_path)
    artifact.add_file(disc_final_path)
    wb_logger.run.log_artifact(artifact)
    logger.info("Model Artifacts logged to WandB Artifact Registry")

    logger.info("Plotting the Loss Graphs")
    generate_plots("results", "results/improved")
    logger.info(f"Finished the Project: {config.project_name} ")
    wb_logger.finish()

if __name__ == "__main__":
    main()
