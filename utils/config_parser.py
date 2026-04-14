import wandb
import yaml
import argparse
import os
ARCHITECTURE_MAP = {
    "shallow": {"g": [256, 128, 64], "d": [64, 128, 256]},
    "standard": {"g": [512, 256, 128, 64], "d": [64, 128, 256, 512]},
    "deep": {"g": [1024, 512, 256, 128, 64], "d": [64, 128, 256, 512, 1024]}
}
class Config:
    """This is a Global class holds the configuration parameters"""
    def __init__(self, config_path):
        with open(config_path,'r') as f:
            self.config = yaml.safe_load(f)
        self.project_name = self.config.get('project_name','GAN_Scientific_Computing')
        self.device = self.config.get('device','cpu')
        self.dataset = self.config.get('dataset', {})
        self.model = self.config.get('model', {})
        self.trainer = self.config.get('trainer', {})
        self.logger = self.config.get('logger', {})
        self.data_transform = self.config.get('data_transform', {})
        self.use_pin = self.config.get('use_pin', False)
        self.data = self.config.get('data', {})
        # Overriding the params for wandb
        if os.environ.get("WANDB_SWEEP_ACTIVE") == "True" and wandb.run is not None:
            sweep_config = wandb.config

            if hasattr(sweep_config,'lr'):
                self.trainer['lr_g'] = sweep_config.lr
                self.trainer['lr_d'] = sweep_config.lr

                # Override Latent and Embedding Dimensions
            if hasattr(sweep_config, 'latent_dim'):
                self.model['latent_dim'] = sweep_config.latent_dim
            if hasattr(sweep_config, 'embedding_dim'):
                self.model['embedding_dim'] = sweep_config.embedding_dim

                # Override Batch size and Epochs
            if hasattr(sweep_config, 'batch_size'):
                self.trainer['batch_size'] = sweep_config.batch_size
            if hasattr(sweep_config, 'epochs'):
                self.trainer['epochs'] = sweep_config.epochs

                # Handle Architecture Override
                # We look for 'architecture' in the sweep and map it to g_channels/d_channels
            if hasattr(sweep_config, 'architecture') and sweep_config.architecture in ARCHITECTURE_MAP:
                arch = ARCHITECTURE_MAP[sweep_config.architecture]
                self.model['g_channels'] = arch['g']
                self.model['d_channels'] = arch['d']

    def __repr__(self):
        return str(self.config)

def get_args():
    """Reads the config file name from the terminal command"""
    parser = argparse.ArgumentParser(description="GAN Training")
    parser.add_argument('--config', type=str, default='baseline_config.yaml',
                        help='Name of the config file in the config/ folder')
    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    return config_path