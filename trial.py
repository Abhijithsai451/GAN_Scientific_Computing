from utils.config_parser import Config, get_args
from utils.wandb_config import WandBConfig

def main():
    # Importing the config file from the command line
    config_path = get_args()

    # Setting up the WandB Logger
    wb_logger = WandBConfig(config_path)
    config =  wb_logger.get_config()

if __name__ == '__main__':
    main()