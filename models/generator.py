import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        arch = config.model
        self.latent_dim = arch['latent_dim']
        # Class Embeddings: Turns the labels into a dense vector
        embedding_dim = arch['embedding_dim']
        self.num_classes = arch['num_classes']
        self.label_embedding = nn.Embedding(self.num_classes,embedding_dim)

        # Initial Block: (latent_dim + Embeddings) --> 256 * 4 * 4
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim,self.label_embedding,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(arch['generator']['initial_channels']),
            nn.ReLU(True)
        )

        # Hidden Layers
        layers = []
        current_channel = arch['generator']['initial_channels']
        channels = arch['generator']['channel_sequence']

        for next_channel in channels:
            layers.append(nn.ConvTranspose2d(current_channel,next_channel,kernel_size=4,stride=2,padding=1))
            layers.append(nn.BatchNorm2d(next_channel))
            layers.append(nn.ReLU(True))
            current_channel = next_channel

        layers.append(nn.ConvTranspose2d(current_channel,out_channels = 3,kernel_size=4,stride=2,padding=1))
        layers.append(nn.Tanh()) # Here the activation function ensures the output is in range of [-1,1]

        self.generator = nn.Sequential(*layers)

    def forward(self, input, labels):
        x = input.view(-1, self.latent_dim, 1,1)
        label = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, label],1)

        x = self.init_block(x)
        gen_images = self.generator(x)
        return gen_images




