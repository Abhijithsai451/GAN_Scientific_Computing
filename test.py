import os

import torch
from torchvision.utils import save_image

from data_processing.dataloader import get_dataloaders
from models.discriminator import Discriminator
from models.generator import Generator
from utils.config_parser import get_args, Config
from utils.logger_config import setup_logger


def test_model():
    config_path = get_args()
    config = Config(config_path)
    logger = setup_logger(config)

    DEVICE = config.device if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Testing the Project: {config.project_name} on {DEVICE} ")
    logger.info(f"Configuration Loaded from {config_path} ")

    generator = Generator(config).to(DEVICE)
    discriminator = Discriminator(config).to(DEVICE)

    # Define checkpoint paths (assumes training has finished)
    ckpt_dir = config.logger.get('ckpt_dir', 'results/checkpoints')
    gen_path = os.path.join(ckpt_dir, "generator_final.pth")
    disc_path = os.path.join(ckpt_dir, "discriminator_final.pth")

    if not os.path.exists(gen_path):
        logger.info(f"Error: Checkpoints not found at {gen_path}. Run training first.")
        return

    generator.load_state_dict(torch.load(gen_path, map_location=DEVICE))
    discriminator.load_state_dict(torch.load(disc_path, map_location=DEVICE))
    generator.eval()
    discriminator.eval()

    _ , test_loader, _ = get_dataloaders(config)

    logger.info("\n>>> Evaluating the Discriminator on the Real (Test Set) vs Fake Data...")
    correct_real = 0
    correct_fake = 0
    total_samples = 0

    with torch.no_grad():
        for i, (real_imgs, labels) in enumerate(test_loader):
            real_imgs, labels = real_imgs.to(DEVICE), labels.to(DEVICE)
            batch_size = real_imgs.size(0)

            # Predict Real
            output_real = torch.sigmoid(discriminator(real_imgs, labels))
            correct_real += (output_real > 0.5).sum().item()

            # Predict Fake
            noise = torch.randn(batch_size, config.model['latent_dim']).to(DEVICE)
            fake_imgs = generator(noise, labels)
            output_fake = torch.sigmoid(discriminator(fake_imgs, labels))
            correct_fake += (output_fake < 0.5).sum().item()

            total_samples += batch_size
            if i >= 10: break

    real_acc = (correct_real / total_samples) * 100
    fake_acc = (correct_fake / total_samples) * 100
    logger.info(f"Accuracy on Real Test Images: {real_acc:.2f}%")
    logger.info(f"Accuracy on Generated Fake Images: {fake_acc:.2f}%")
    print("\n>>> Generating Test Grid for All Classes...")
    os.makedirs("results/test_results", exist_ok=True)

    # Creating 8 samples per class for all 10 classes
    num_classes = config.model['num_classes']
    test_labels = torch.arange(num_classes).repeat_interleave(8).to(DEVICE)
    test_noise = torch.randn(len(test_labels), config.model['latent_dim']).to(DEVICE)

    with torch.no_grad():
        final_samples = generator(test_noise, test_labels)
        save_path = f"results/test_results/final_test_grid_{config.project_name}.png"
        save_image(final_samples, save_path, nrow=8, normalize=True)

    print(f"Final test grid saved to: {save_path}")
    print("--- Testing Complete ---")


if __name__ == "__main__":
    test_model()