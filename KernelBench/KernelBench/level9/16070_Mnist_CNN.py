import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class Mnist_CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


def get_inputs():
    return [torch.rand([4, 1, 28, 28])]


def get_init_inputs():
    return [[], {}]
