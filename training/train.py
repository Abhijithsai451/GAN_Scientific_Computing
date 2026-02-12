import torch
import torch.nn as nn
from jinja2 import optimizer

from data_processing import dataloader
from data_processing.data_transformer import device
from models import generator, discriminator

criterion = nn.BCELoss()
gen_optim = torch.optim.Adam(generator.parameters(), lr=0.001,betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001,betas=(0.5, 0.999))
num_epochs = 100
REAL_IMAGE = 1.0
FAKE_IMAGE = 0.0

for epoch in range(num_epochs):
    for i, (image, labels) in enumerate(dataloader):
        batch_size = image.size(0)
        images = images.to(device)
        labels = labels.to(device)

        # Training the Discriminator
        optimizer.zero_grad()

        output_real = discriminator(images, labels)
        label_real = torch.full((batch_size, 1), REAL_IMAGE, device=device)
        loss_D_real = criterion(output_real, label_real)

        # Part B: Train with Fake Images
        noise = torch.randn(batch_size, 100, device=device)  # Latent vector z [cite: 81]
        fake_images = generator(noise, labels)
        output_fake = discriminator(fake_images.detach(), labels)
        label_fake = torch.full((batch_size, 1), FAKE_IMAGE, device=device)
        loss_D_fake = criterion(output_fake, label_fake)

        # Combined Loss and Update
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        disc_optim.step()

        gen_optim.zero_grad()

        # The Generator wants the Discriminator to think these are REAL
        output_fake_for_G = discriminator(fake_images, labels)
        loss_G = criterion(output_fake_for_G, label_real)  # Uses REAL labels to "fool" D

        loss_G.backward()
        gen_optim.step()