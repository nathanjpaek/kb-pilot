import torch
import torch.nn as nn


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)


class GeneratorBlock(nn.Module):

    def __init__(self, input_channels, latent_channels, output_channels,
        upsample=True):
        super(GeneratorBlock, self).__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)
        else:
            self.upsample = nn.Identity()
        self.conv1 = nn.Conv2d(input_channels, latent_channels, 3, 1, 1,
            padding_mode='replicate')
        self.conv1.bias.data = torch.zeros(*self.conv1.bias.data.shape)
        self.act1 = leaky_relu()
        self.conv2 = nn.Conv2d(latent_channels, output_channels, 3, 1, 1,
            padding_mode='replicate')
        self.conv2.bias.data = torch.zeros(*self.conv2.bias.data.shape)
        self.act2 = leaky_relu()
        self.to_rgb = nn.Conv2d(output_channels, 3, 1, 1, 0)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        rgb = self.to_rgb(x)
        return x, rgb


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'latent_channels': 4,
        'output_channels': 4}]
