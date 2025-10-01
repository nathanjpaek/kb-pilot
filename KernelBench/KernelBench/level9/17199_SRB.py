import torch
import torch.nn as nn
import torch.utils.model_zoo


class SRB(nn.Module):

    def __init__(self):
        super(SRB, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2)
        self.act = nn.ReLU(True)

    def forward(self, x):
        o1 = self.act(self.conv1(x))
        o2 = self.act(self.conv2(o1))
        o3 = self.conv3(o2)
        return o3


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
