import torch
from torch import nn
from torch.autograd import Variable


def add_gaussian_noise(x, std):
    return x + Variable(x.data.new(x.size()).normal_(0, std))


class CDAE(nn.Module):
    """
    Convolutional denoising autoencoder layer for stacked autoencoders.

    Args:
        in_channels: the number of channels in the input.
        out_channels: the number of channels in the output.
        stride: stride of the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=2,
        noise_std=0.1, **kwargs):
        super(CDAE, self).__init__(**kwargs)
        self.std = noise_std
        self.encoder = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=0)
        self.decoder = nn.ConvTranspose2d(out_channels, in_channels,
            kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if self.training:
            x += add_gaussian_noise(x, self.std)
        emb = torch.relu(self.encoder(x))
        return emb, torch.relu(self.decoder(emb))

    def reconstruct(self, emb):
        return self.decoder(emb)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
