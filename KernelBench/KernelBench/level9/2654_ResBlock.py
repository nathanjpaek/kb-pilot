import torch
from torch import nn
from torchvision import models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx


class ResBlock(nn.Module):

    def __init__(self, num_of_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of_channels, out_channels=
            num_of_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(num_of_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_of_channels, out_channels=
            num_of_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(num_of_channels, affine=True)

    def forward(self, x):
        orig = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, orig)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_of_channels': 4}]
