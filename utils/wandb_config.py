import wandb
import torch
import os

import yaml


class WandBConfig:
    def __init__(self, config_file, job_type="train"):
        with open(config_file,'r') as f:
            config_data = yaml.safe_load(f)
        self.config = config_data

        self.run = wandb.init(
            project=self.config['project_name'],
            config= self.config,
            job_type=job_type,
        )

    def get_config(self):
        return wandb.config