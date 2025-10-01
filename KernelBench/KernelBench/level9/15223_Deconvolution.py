import torch
import torch.nn as nn
import torch.utils.model_zoo


class Deconvolution(nn.Module):

    def __init__(self, C, stride):
        super(Deconvolution, self).__init__()
        if stride == 2:
            kernel_size = 3
            output_padding = 1
        elif stride == 4:
            kernel_size = 5
            output_padding = 1
        else:
            kernel_size = 3
            output_padding = 0
        self.deconv = nn.ConvTranspose2d(C, C, kernel_size=kernel_size,
            stride=stride, padding=1, output_padding=output_padding)

    def forward(self, x):
        return self.deconv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C': 4, 'stride': 1}]
