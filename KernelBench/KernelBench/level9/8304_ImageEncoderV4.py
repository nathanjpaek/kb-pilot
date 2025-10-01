import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoderV4(nn.Module):
    """
    Outputs a 5 x 5 x 32 feature map that preserves spatial information.
    """

    def __init__(self, input_channels=3, init_scale=1.0, no_weight_init=
        False, init_method='ortho', activation='relu'):
        super(ImageEncoderV4, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.conv3):
                if init_method == 'ortho':
                    nn.init.orthogonal_(layer.weight, init_scale)
                elif init_method == 'normal':
                    nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight, 1.0)
                else:
                    assert init_method == 'default'

    def forward(self, imgs):
        if self.activation == 'relu':
            ac = F.relu
        elif self.activation == 'tanh':
            ac = torch.tanh
        else:
            raise RuntimeError()
        x = ac(self.conv1(imgs))
        x = ac(self.conv2(x))
        x = ac(self.conv3(x))
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
