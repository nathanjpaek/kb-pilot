import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


class TReLU(nn.Module):

    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = weightNorm(nn.Conv2d(6, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 1, 1, 1, 0))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = x.view(-1, 64)
        return x


def get_inputs():
    return [torch.rand([4, 6, 64, 64])]


def get_init_inputs():
    return [[], {}]
