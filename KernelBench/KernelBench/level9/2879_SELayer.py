import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms.functional as F
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/    0344c5503ee55e24f0de7f37336a6e08f10976fd/    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SELayer(nn.Module):
    REDUCTION = 4

    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.channel = channel
        self.reduction = SELayer.REDUCTION
        num_mid = make_divisible(self.channel // self.reduction, divisor=8)
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
