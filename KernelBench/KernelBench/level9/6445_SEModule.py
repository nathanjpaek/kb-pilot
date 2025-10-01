import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.data


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, reduction=0.25):
        super(SEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        num_mid = make_divisible(int(self.channel * self.reduction), divisor=8)
        self.fc = nn.Sequential(OrderedDict([('reduce', nn.Conv2d(self.
            channel, num_mid, 1, 1, 0, bias=True)), ('relu', nn.ReLU(
            inplace=True)), ('expand', nn.Conv2d(num_mid, self.channel, 1, 
            1, 0, bias=True)), ('h_sigmoid', Hsigmoid(inplace=True))]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
