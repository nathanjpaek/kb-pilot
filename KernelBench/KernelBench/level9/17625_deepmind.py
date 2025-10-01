import torch
import torch.nn as nn
import torch.nn.functional as F


class deepmind(nn.Module):

    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        return x


def get_inputs():
    return [torch.rand([4, 4, 144, 144])]


def get_init_inputs():
    return [[], {}]
