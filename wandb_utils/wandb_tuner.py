import os
import uuid

import yaml
import subprocess
import wandb
import pandas as pd


class HyperParameterTuner:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.dirname(self.script_dir)
        self.config_dir = os.path.join(self.root, "config")
        self.arch_map = {
            "shallow": {"g": [256, 128, 64], "d": [64, 128, 256]},
            "standard": {"g": [512, 256, 128, 64], "d": [64, 128, 256, 512]},
            "deep": {"g": [1024, 512, 256, 128, 64], "d": [64, 128, 256, 512, 1024]}
        }

    def sweep_worker(self):
        with wandb.init() as run:
            swp = wandb.config
            trial_name = f"sweep_{run.id}"

            # Map the W&B architecture string to the channel lists
            arch = self.arch_map[swp.architecture]

            # Prepare the parameters in the format your run_experiment expects
            params = {
                'lr': swp.lr,
                'latent_dim': swp.latent_dim,
                'embedding_dim': swp.embedding_dim,
                'g_channels': arch['g'],
                'd_channels': arch['d'],
                'leaky_slope': 0.2,
                'beta1': 0.5,
                'beta2': 0.999
            }

        # Use your existing experiment runner!
        self.run_experiment(params, trial_name)

    def run_experiment(self, params, name):
        """Your existing experiment runner (Minimal changes)"""
        temp_file_path = os.path.join(self.config_dir, f"temp_{name}.yaml")
        base_path = os.path.join(self.config_dir, 'improved_config.yaml')

        with open(base_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update config with sweep params
        config['project_name'] = name
        config['trainer'].update({'lr_g': params['lr'], 'lr_d': params['lr'], 'epochs': 2})
        config['model'].update({
            'latent_dim': params['latent_dim'],
            'embedding_dim': params['embedding_dim'],
            'generator_channels': params['g_channels'],
            'discriminator_channels': params['d_channels']
        })

        try:
            with open(temp_file_path, 'w') as f:
                yaml.dump(config, f)

            print(f"\n[MLOps Tuner] Running Trial: {name}")
            # Runs main.py exactly as your shell script does
            env = os.environ.copy()
            for key in list(env.keys()):
                if key.startswith("WANDB_"):
                    del env[key]
            if "wandb_api_key" in env:
                env["WANDB_API_KEY"] = env["wandb_api_key"]

            subprocess.run(["python", "trial.py", "--config", temp_file_path], env=env)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def run_wandb_sweep(self, sweep_config_path, count=10):
        """Initializes and starts the W&B Sweep"""
        with open(sweep_config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project="GAN_Scientific_Computing")
        # This starts the agent which calls sweep_worker multiple times
        wandb.agent(sweep_id, function=self.sweep_worker, count=count)


if __name__ == "__main__":
    tuner = HyperParameterTuner()
    # Replace manual grid search with the MLOps sweep
    tuner.run_wandb_sweep("config/sweep_config.yaml", count=10)