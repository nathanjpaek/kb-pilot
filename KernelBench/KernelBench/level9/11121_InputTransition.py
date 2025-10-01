import torch
import torch.nn as nn


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class InputTransition(nn.Module):

    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(32)
        self.relu1 = ELUCons(elu, 32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(32)
        self.relu2 = ELUCons(elu, 32)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {'outChans': 4, 'elu': 4}]
