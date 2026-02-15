import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        arch = config.model
        self.num_classes = arch.get('num_classes',10)
        self.embedding_dim = arch['embedding_dim']
        self.label_embedding = nn.Embedding(self.num_classes, self.embedding_dim)

        channels_sequence = arch['discriminator_channels']
        self.leaky_slope = arch.get('leaky_slope',0.2)

        layers = []
        in_channels = 3

        for i, out_channels in enumerate(channels_sequence):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias = False)
            )
            if i > 0:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(self.leaky_slope, inplace=False))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        last_out_channels = arch['discriminator_channels'][-1]

        self.classifier = nn.Sequential(
            nn.Linear(last_out_channels + self.embedding_dim, 1),
        )
    def forward(self, img, labels):
        x = self.features(img)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        c = self.label_embedding(labels)
        combined = torch.cat((x, c), 1)
        validity = self.classifier(combined)
        return validity