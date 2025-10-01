import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):

    def forward(self, x, y):
        return F.cross_entropy(x, torch.argmax(y, -1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
