import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, input_channels: 'int', output_channels: 'int',
        batch_norm=False) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size
            =3, stride=1, padding=1)
        self.bn1 = nn.Identity()
        self.conv2 = nn.Conv2d(output_channels, output_channels,
            kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.Identity()
        self.conv_skip = nn.Conv2d(input_channels, output_channels,
            kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(output_channels)
            self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        skip_y = self.conv_skip(x)
        y = y + skip_y
        y = self.relu2(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
