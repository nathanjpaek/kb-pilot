import torch
import torch.nn as nn


class UpConv(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size):
        super(UpConv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=input_nc, out_channels
            =output_nc, kernel_size=2, bias=True, stride=2, padding=0)
        self.activation_fn = nn.ELU()

    def forward(self, input):
        return self.activation_fn(self.deconv(input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nc': 4, 'output_nc': 4, 'kernel_size': 4}]
