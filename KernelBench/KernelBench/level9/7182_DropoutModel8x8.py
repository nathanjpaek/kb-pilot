import torch
import torch.nn as nn
import torch.nn.functional as func


class DropoutModel8x8(nn.Module):

    def __init__(self, channel):
        """
        Define useful layers
        Argument:
        channel: number of channel, or depth or number of different sprite types
        """
        super(DropoutModel8x8, self).__init__()
        self.dropout_1 = nn.Dropout2d(0.3)
        self.conv_1 = nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv2d(channel * 2, channel * 4, kernel_size=3,
            stride=1)
        self.conv_3 = nn.Conv2d(channel * 4, channel * 8, kernel_size=3,
            stride=1)
        self.conv_middle = nn.Conv2d(channel * 8, channel * 8, kernel_size=
            3, stride=1, padding=1)
        self.conv_T1 = nn.ConvTranspose2d(channel * 8, channel * 4,
            kernel_size=3, stride=1)
        self.conv_T2 = nn.ConvTranspose2d(channel * 4, channel * 2,
            kernel_size=3, stride=1)
        self.conv_T3 = nn.ConvTranspose2d(channel * 2, channel, kernel_size
            =3, stride=1)

    def forward(self, x):
        if self.training:
            x = self.dropout_1(x)
        x = func.relu(self.conv_1(x))
        x = func.relu(self.conv_2(x))
        x = func.relu(self.conv_3(x))
        x = self.conv_middle(x)
        x = self.conv_T1(x)
        x = self.conv_T2(x)
        x = torch.sigmoid(self.conv_T3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'channel': 4}]
