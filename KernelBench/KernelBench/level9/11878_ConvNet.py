import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """ convolutional neural network """

    def __init__(self):
        super(ConvNet, self).__init__()
        nf = 8
        self.conv1 = nn.Conv2d(1, nf * 1, 5, 1, 0)
        self.conv2 = nn.Conv2d(nf * 1, nf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 5, 1, 0)
        self.conv4 = nn.Conv2d(nf * 4, nf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(nf * 8, 10, 4, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
