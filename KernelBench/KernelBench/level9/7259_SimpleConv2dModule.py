import torch
import torch.nn.functional as F
import torch.jit
import torch.onnx
import torch.nn


class SimpleConv2dModule(torch.nn.Module):

    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        super(SimpleConv2dModule, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, inputs, filters, bias=None):
        conv = F.conv2d(inputs, filters, bias=bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        return F.relu(conv)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
