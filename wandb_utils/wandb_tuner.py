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
            #"standard": {"g": [512, 256, 128, 64], "d": [64, 128, 256, 512]},
            #"deep": {"g": [1024, 512, 256, 128, 64], "d": [64, 128, 256, 512, 1024]}
        }
        self.results = []
        self.best_config = None

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

        result_entry = {**params, 'final_g_loss': 0.0}
        self.results.append(result_entry)
        self.best_config = params

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

    def run_wandb_sweep(self, sweep_config_path,):
        """Initializes and starts the W&B Sweep"""
        with open(sweep_config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project="GAN_Scientific_Computing")
        wandb.agent(sweep_id, function=self.sweep_worker)

    def save_best_config(self, target_config_name="improved_config.yaml"):
        if not self.best_config:
            print("No best configuration found to save.")
            return

        target_path = os.path.join(self.config_dir, target_config_name)

        with open(target_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update with best parameters
        print(f"\n>>> Overwriting {target_config_name} with best parameters...")
        config['trainer']['lr_g'] = self.best_config['lr']
        config['trainer']['lr_d'] = self.best_config['lr']
        config['model']['latent_dim'] = self.best_config['latent_dim']
        config['model']['embedding_dim'] = self.best_config['embedding_dim']
        config['model']['generator_channels'] = self.best_config['g_channels']
        config['model']['discriminator_channels'] = self.best_config['d_channels']

        class CleanDumper(yaml.SafeDumper):
            def represent_data(self, data):
                if isinstance(data, list):
                    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
                return super().represent_data(data)

        with open(target_path, 'w') as f:
            yaml.dump(config, f, Dumper=CleanDumper, sort_keys=False, default_flow_style=False)
        print(f">>> {target_config_name} updated successfully.")
    def print_summary(self):
        print("\n" + "=" * 50)
        print("TUNING SUMMARY")
        print("=" * 50)
        for i, res in enumerate(self.results):
            print(
                f"Trial {i}: G_Loss: {res.get('final_g_loss', 'N/A'):.4f} | LR: {res['lr']} | Slope: {res['leaky_slope']}")

        if self.best_config:
            print("\n>>> BEST CONFIGURATION FOUND:")
            print(self.best_config)
        print("=" * 50)
if __name__ == "__main__":
    tuner = HyperParameterTuner()
    # Replace manual grid search with the MLOps sweep
    tuner.run_wandb_sweep("config/sweep_config.yaml")
    tuner.print_summary()
    tuner.save_best_config()