import torch
from torch.nn import functional as F
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 50, kernel_size=3)
        self.conv4 = nn.Conv2d(50, 2, kernel_size=1, bias=False, padding=0,
            stride=1)
        self.max_pool2d = nn.MaxPool2d((4, 4))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2d(x)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
