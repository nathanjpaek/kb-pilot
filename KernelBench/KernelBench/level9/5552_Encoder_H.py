import torch
import torch.nn as nn


class Encoder_H(nn.Module):

    def __init__(self, input_shape=(64, 64), z_dim=10, nc=3, padding=1):
        super(Encoder_H, self).__init__()
        self.conv2d_1 = nn.Conv2d(nc, 32, 4, 2, padding)
        self.conv2d_2 = nn.Conv2d(32, 32, 4, 2, padding)
        self.conv2d_3 = nn.Conv2d(32, 64, 4, 2, padding)
        self.conv2d_4 = nn.Conv2d(64, 64, 4, 2, padding)
        self.flatten_shape, self.dconv_size = self._get_conv_output(input_shape
            , nc)
        self.linear = nn.Linear(self.flatten_shape, z_dim * 2)

    def _get_conv_output(self, shape, nc):
        bs = 1
        dummy_x = torch.empty(bs, nc, *shape)
        x, dconv_size = self._forward_features(dummy_x)
        flatten_shape = x.flatten(1).size(1)
        return flatten_shape, dconv_size

    def _forward_features(self, x):
        size0 = x.shape[1:]
        x = torch.relu(self.conv2d_1(x))
        size1 = x.shape[1:]
        x = torch.relu(self.conv2d_2(x))
        size2 = x.shape[1:]
        x = torch.relu(self.conv2d_3(x))
        size3 = x.shape[1:]
        x = torch.relu(self.conv2d_4(x))
        size4 = x.shape[1:]
        return x, [size0, size1, size2, size3, size4]

    def forward(self, x):
        x = torch.relu(self.conv2d_1(x))
        x = torch.relu(self.conv2d_2(x))
        x = torch.relu(self.conv2d_3(x))
        x = torch.relu(self.conv2d_4(x))
        x = self.linear(x.flatten(1))
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
