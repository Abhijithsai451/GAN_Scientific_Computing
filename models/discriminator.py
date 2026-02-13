import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        arch = config.model
        self.label_embeddings = nn.Embedding(arch['num_classes'], arch['embedding_dim'])
        channel_sequence = arch['discriminator']['channel_sequence']
        self.leaky_slope = arch['discriminator']['leaky_slope']
        layers = []
        in_channels = 3
        for i, out_channels in enumerate(channel_sequence):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            if i > 0 :
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(negative_slope= self.leaky_slope, inplace=True))
            in_channels = out_channels

        self.block = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_channels * 4 * 4 + arch['embedding_dim'],out_features= 1),
            nn.Sigmoid()
        )
    def forward(self,img, labels):
        x = self.block(img)
        x = torch.flatten(x, 1)
        c = self.label_embeddings(labels)
        combined = torch.cat((x, c), dim=1)
        validity = self.classifier(combined)
        return validity
