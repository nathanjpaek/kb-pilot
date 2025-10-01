import torch
import torch.nn as nn


class ExpanderConv2d(nn.Module):

    def __init__(self, indim, outdim, kernel_size, expandSize, stride=1,
        padding=0, inDil=1, groups=1, mode='random'):
        super(ExpanderConv2d, self).__init__()
        self.conStride = stride
        self.conPad = padding
        self.outPad = 0
        self.conDil = inDil
        self.conGroups = groups
        self.bias = True
        self.weight = torch.nn.Parameter(data=torch.Tensor(outdim, indim,
            kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_normal_(self.weight.data, mode='fan_out')
        self.mask = torch.zeros(outdim, indim, 1, 1)
        if indim > outdim:
            for i in range(outdim):
                x = torch.randperm(indim)
                for j in range(expandSize):
                    self.mask[i][x[j]][0][0] = 1
        else:
            for i in range(indim):
                x = torch.randperm(outdim)
                for j in range(expandSize):
                    self.mask[x[j]][i][0][0] = 1
        self.mask = self.mask.repeat(1, 1, kernel_size, kernel_size)
        self.mask = nn.Parameter(self.mask)
        self.mask.requires_grad = False

    def forward(self, dataInput):
        extendWeights = self.weight.clone()
        extendWeights.mul_(self.mask.data)
        return torch.nn.functional.conv2d(dataInput, extendWeights, bias=
            None, stride=self.conStride, padding=self.conPad, dilation=self
            .conDil, groups=self.conGroups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'indim': 4, 'outdim': 4, 'kernel_size': 4, 'expandSize': 4}]
