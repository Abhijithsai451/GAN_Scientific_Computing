import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, embedding_dim):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.label_embedding = nn.Embedding(n_classes, embedding_dim)
        self.init_layer = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 512 * 4 * 4),
            nn.ReLU(True),
        )
        self.upsample = nn.Sequential(
            #1. Input: 512 x 4 x 4 --> Output: 256 x 8 x 8 (Latent Dimension to the Conv Output)
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 1. Input: 256 x 8 x 8 --> Output: 128 x 16 x 16 (Layer 1  to the Conv Output)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Final Layer
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Tanh()
        )

        def forward(self,noise,  x):
            label_embedding = self.label_embedding(x)
            gen_input = torch.cat((noise, label_embedding), dim=1)
            out = self.init_layer(gen_input)
            out = out.view(-1, 512,4,4)
            img = self.upsample(out)
            return img