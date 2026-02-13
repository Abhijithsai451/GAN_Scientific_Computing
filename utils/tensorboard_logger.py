import os

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, config):
        self.log_dir = config.logger['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir,'history.csv')
        self.history = []

        self.writer = None
        if config.logger.get('use_tensorboard', False):
            self.writer = SummaryWriter(log_dir= os.path.join(self.log_dir, 'tensorboard'))

    def log_epoch(self, epoch, loss_d, loss_g, d_x, d_gz):
        self.history.append({
            'epoch': epoch,
            'loss_d': loss_d,
            'loss_g': loss_g,
            'd_x': d_x,
            'd_gz': d_gz
        })
        if self.writer:
            self.writer.add_scalar('Loss/Discriminator', loss_d, epoch)
            self.writer.add_scalar('Loss/Generator', loss_g, epoch)
            self.writer.add_scalars('Probabilities', {"Real": d_x, "Fake": d_gz}, epoch)

        def save_to_csv(self):
            df = pd.DataFrame(self.history)
            df.to_csv(self.csv_path, index=False)

        def close(self):
            if self.writer:
                self.writer.close()

