import torch
import torch.nn as nn
import torch.nn.functional as F


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(1.2 * x + 3.0, inplace=self.inplace) / 6.0


class Activation(nn.Module):

    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            raise NotImplementedError
        elif act_type == 'hard_sigmoid':
            self.act = Hsigmoid(inplace)
        elif act_type == 'hard_swish':
            self.act = Hswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)


class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4, name=''):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=
            in_channels // reduction, kernel_size=1, stride=1, padding=0,
            bias=True)
        self.relu1 = Activation(act_type='relu', inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction,
            out_channels=in_channels, kernel_size=1, stride=1, padding=0,
            bias=True)
        self.hard_sigmoid = Activation(act_type='hard_sigmoid', inplace=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hard_sigmoid(outputs)
        outputs = inputs * outputs
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
