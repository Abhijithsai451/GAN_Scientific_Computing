import wandb
import yaml
import os

ARCHITECTURE_MAP = {
    "shallow": {"g": [256, 128, 64], "d": [64, 128, 256]},
    "standard": {"g": [512, 256, 128, 64], "d": [64, 128, 256, 512]},
    "deep_1": {"g": [512, 256, 128, 128, 128, 64], "d": [64, 128, 128, 128, 256, 512]},
    "deep": {"g": [1024, 512, 256, 128, 64], "d": [64, 128, 256, 512, 1024]}
}
def run_tuner():
    with wandb.init() as run:
        sweep_config = wandb.config

if __name__ == "__main__":
    run_tuner()