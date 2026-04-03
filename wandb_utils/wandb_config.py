import wandb
import os
import yaml

class WandBConfig:
    def __init__(self, config_file, job_type="train"):
        with open(config_file,'r') as f:
            config_data = yaml.safe_load(f)
        self.config = config_data
        is_sweep = "WANDB_SWEEP_ID" in os.environ
        if not is_sweep:
            project_name = self.config.get('project_name',"GAN_Scientific_Computing")
        self.run = wandb.init(
            config= self.config,
            job_type=job_type,
            mode="offline",
            settings=wandb.Settings(init_timeout=300)
        )

    def get_config(self):
        return wandb.config

    def log_step(self,metrics, step = None):
        wandb.log(metrics, step = step)

    def log_images(self, images, title="Generated Samples", step = None):
        wandb.log({title: [wandb.Image(img) for img in images]}, step=step)

    def finish(self):
        wandb.finish()