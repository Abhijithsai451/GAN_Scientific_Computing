import logging
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

_writer = None
_project_name = None
def setup_logger(config):
    """
    Sets up logger for console output and Training Output
    """
    global _project_name
    _project_name = config.project_name
    logger = logging.getLogger(_project_name)

    if logger.hasHandlers():
        return logger

    log_dir = config.logger["log_dir"]
    log_file = config.logger["log_file"]
    log_path = os.path.join(log_dir, log_file)

    os.makedirs(log_dir, exist_ok=True)

    # Use the configuration you provided
    logging.basicConfig(
        level=getattr(logging, config.logger['level']),
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger.info(f"Logger identity set to: {_project_name}")
    return logger


def get_logger():
    """
    Retrieves the global project logger instance.
    Can be called in any file without needing the config object.
    """
    global _project_name
    # If setup_logger hasn't been called, it defaults to the root logger
    return logging.getLogger(_project_name)


def tensorboard_logger(config):
    """
    Initializes the Tensorboard writer.
    """
    global _writer
    if _writer is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(config.logger["tensorboard_dir"], run_name)
        os.makedirs(log_dir, exist_ok=True)
        _writer = SummaryWriter(log_dir=log_dir)
    return _writer