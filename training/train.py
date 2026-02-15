import os

import torch
from torch import nn


class GANTrainer:
    def __init__(self, generator, discriminator, config, device):
        self.gen = generator
        self.disc = discriminator
        self.config = config
        self.device = device

        beta1 = float(config.trainer.get('beta1'))
        beta2 = float(config.trainer.get('beta2'))
        # Loss function & Optimizers
        self.criterion = nn.BCEWithLogitsLoss()
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=config.trainer['lr_g'],
                                      betas = (beta1, beta2))
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=config.trainer['lr_d'],
                                      betas = (beta1, beta2))
        self.static_real_label = torch.ones(config.trainer['batch_size'], 1).to(self.device)
        self.static_fake_label = torch.zeros(config.trainer['batch_size'], 1).to(self.device)

    def train_step(self, real_imgs, labels):
        batch_size = real_imgs.size(0)

        real_label = self.static_real_label[:batch_size]
        fake_label = self.static_fake_label[:batch_size]
        # Train Discriminator
        self.opt_d.zero_grad(True)

        # Real images
        output_real = self.disc(real_imgs, labels)
        loss_d_real = self.criterion(output_real, real_label)

        # Fake images
        noise = torch.randn(batch_size, self.config.model['latent_dim']).to(self.device)
        fake_imgs = self.gen(noise, labels)
        output_fake = self.disc(fake_imgs.detach(), labels)
        loss_d_fake = self.criterion(output_fake, fake_label)

        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        self.opt_d.step()

        # 2. Train Generator
        self.opt_g.zero_grad(True)

        # We want the discriminator to think the fakes are real
        output_gen = self.disc(fake_imgs, labels)
        loss_g = self.criterion(output_gen, real_label)

        loss_g.backward()
        self.opt_g.step()

        return loss_d.item(), loss_g.item(), output_real.mean().item(), output_fake.mean().item()
        #return 0.0, 0.0, 0.5, 0.5

    def save_checkpoint(self, epoch, path="results/checkpoints"):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'gen_state_dict': self.gen.state_dict(),
            'disc_state_dict': self.disc.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(path, f"ckpt_epoch_{epoch}.pth"))