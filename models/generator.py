import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        arch = config.model

        self.latent_dim = arch['latent_dim']
        self.embedding_dim = arch['embedding_dim']
        self.num_classes = arch.get('num_classes',10)
        self.label_embedding = nn.Embedding(self.num_classes,self.embedding_dim)
        self.initial_dim = self.latent_dim + self.embedding_dim

        channels_list = arch['generator_channels']

        # Initial Block: (latent_dim + Embeddings) --> 256 x 4 x 4
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = self.initial_dim,
                out_channels=channels_list[0],
                kernel_size=4,
                stride=1,
                padding=0,
                bias = False),
            nn.BatchNorm2d(channels_list[0]),
            nn.ReLU(True)
        )

        # Hidden Layers
        layers = []

        for i in range(len(channels_list)-2):
            in_ch = channels_list[i]
            out_ch = channels_list[i+1]
            layers.append(nn.ConvTranspose2d(in_ch,out_ch,kernel_size=4,stride=2,padding=1, bias = False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(channels_list[-2],out_channels = 3,kernel_size=4,stride=2,padding=1))
        layers.append(nn.Tanh()) # Here the activation function ensures the output is in range of [-1,1]

        self.generator = nn.Sequential(*layers)

    def forward(self, input, labels):
        x = input.view(input.size(0), self.latent_dim, 1,1)
        label = self.label_embedding(labels).view(labels.size(0), self.embedding_dim, 1, 1)
        x = torch.cat([x, label],1)

        x = self.init_block(x)
        gen_images = self.generator(x)
        return gen_images




