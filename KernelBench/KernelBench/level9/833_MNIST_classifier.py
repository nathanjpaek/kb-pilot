import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class MNIST_classifier(nn.Module):

    def __init__(self):
        super(MNIST_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv5 = nn.Conv2d(128, 10, 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.conv5(x).view(-1, 10)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
