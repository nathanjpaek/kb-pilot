import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """Simple convolutional autoencoder
    ...
    
    Methods
    -------
    forward(x)
        Forward pass of x
    """

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv_2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pooling_func = nn.MaxPool2d(2, 2)
        self.trans_conv_1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.trans_conv_2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = self.pooling_func(out)
        out = F.relu(self.conv_2(out))
        out = self.pooling_func(out)
        out = F.relu(self.trans_conv_1(out))
        out = torch.sigmoid(self.trans_conv_2(out))
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
