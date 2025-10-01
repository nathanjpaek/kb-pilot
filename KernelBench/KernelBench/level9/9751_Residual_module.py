import torch
import torch.nn as nn


class Residual_module(nn.Module):

    def __init__(self, in_ch):
        super(Residual_module, self).__init__()
        self.prelu1 = nn.PReLU(in_ch, 0)
        self.prelu2 = nn.PReLU(in_ch, 0)
        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
            kernel_size=1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
            kernel_size=1)

    def forward(self, input):
        output_residual = self.conv1_1by1(input)
        output_residual = self.prelu1(output_residual)
        output_residual = self.conv2_1by1(output_residual)
        output = torch.mean(torch.stack([input, output_residual]), dim=0)
        output = self.prelu2(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4}]
