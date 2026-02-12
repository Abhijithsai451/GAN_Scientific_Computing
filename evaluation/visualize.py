import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.logger_config import get_logger

logger = get_logger()

def visualize_batch(dataloader,class_names, num_images= 8,  title="CIFAR 10 Batch"):
    """
    This Function will visualize the batch of CIFAR 10 data we imported using the dataloaders.
    This also denormalization from [-1,1] to [0,1].
    """
    logger.info(f"Visualizing batch of CIFAR 10 data...")

    # Extracting a batch from the dataloader
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Denormalizing the images
    images = images * 0.5 + 0.5
    cols = 4
    rows = (num_images + cols - 1) // cols

    # 4. Setup Plot
    fig = plt.figure(figsize=(12, 3 * rows))
    fig.suptitle(title, fontsize=16)

    for i in range(num_images):
        ax = fig.add_subplot(rows, cols, i + 1)

        # Convert [C, H, W] -> [H, W, C] [cite: 282]
        img_np = images[i].permute(1, 2, 0).cpu().numpy()

        ax.imshow(img_np, interpolation='nearest')

        ax.set_title(f"Class: {class_names[labels[i]]}")
        ax.axis('off')

    plt.tight_layout()
    fig.show()
    logger.info("Interactive visualization rendered.")

def visualize_images(dataloader, config):
    pass
