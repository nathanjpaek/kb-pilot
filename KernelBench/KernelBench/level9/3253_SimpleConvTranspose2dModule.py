import torch
import torch.nn.functional as F
import torch.jit
import torch.onnx
import torch.nn


class SimpleConvTranspose2dModule(torch.nn.Module):

    def __init__(self, stride=1, padding=0, output_padding=0, dilation=1,
        groups=1):
        super(SimpleConvTranspose2dModule, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

    def forward(self, inputs, filters, bias=None):
        convTranspose = F.conv_transpose2d(inputs, filters, bias=bias,
            stride=self.stride, padding=self.padding, output_padding=self.
            output_padding, groups=self.groups, dilation=self.dilation)
        return F.relu(convTranspose)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
