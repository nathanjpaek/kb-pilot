import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion_feature(nn.Module):

    def __init__(self):
        super(Fusion_feature, self).__init__()
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_1x1 = nn.Conv2d(384, 256, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_1x1 = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        conv3 = self.conv3(x)
        conv4 = self.conv4(F.relu(conv3))
        conv5 = F.relu(self.conv5(F.relu(conv4)) + self.conv4_1x1(conv4) +
            self.conv3_1x1(conv3))
        return conv5


def get_inputs():
    return [torch.rand([4, 192, 64, 64])]


def get_init_inputs():
    return [[], {}]
