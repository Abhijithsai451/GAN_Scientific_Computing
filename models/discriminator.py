import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes, embedding_dim):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, embedding_dim)

        self.model = nn.Sequential(
            #1. 3 x 64 x 64 --> 64 x 32 x 32
            nn.Conv2d(3,64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            #2. 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            #3. 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            #4. 512 x 4 x 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final Output Layer
        self.final_layer = nn.Sequential(
            nn.Linear(512 * 4 * 4 + embedding_dim, 1),
            nn.Sigmoid() # Here The ouput we get is a probability, 0 or 1.
        )

    def forward(self, img, labels):
        # Pass image through convolutions
        conv_out = self.model(img)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatens the Image

        label_embedding = self.label_emb(labels)# Concats the flattened image features with class embeddings
        combined = torch.cat((conv_out, label_embedding), dim=1)

        validity = self.final_layer(combined)
        return validity