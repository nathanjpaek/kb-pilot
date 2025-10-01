import torch
import torch.nn as nn


class double_decoder_conv(nn.Module):

    def __init__(self, input_channels1, output_channels1, output_channels2):
        super(double_decoder_conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels1, output_channels1,
            kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(output_channels1, output_channels2,
            kernel_size=3, padding='same')
        self.relu_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.relu_activation(x))
        x = self.conv2(self.relu_activation(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels1': 4, 'output_channels1': 4,
        'output_channels2': 4}]
