import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class CosineClassifier(nn.Module):

    def __init__(self, classes, channels=512):
        super().__init__()
        self.channels = channels
        self.cls = nn.Conv2d(channels, classes, 1, bias=False)
        self.scaler = 10.0

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.scaler * F.conv2d(x, F.normalize(self.cls.weight, dim=1,
            p=2))


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {'classes': 4}]
