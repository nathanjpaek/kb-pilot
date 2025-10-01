import torch
import torch.nn as nn


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):

    def __init__(self, inChans, outChans, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, outChans)
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(outChans)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inChans': 4, 'outChans': 4, 'elu': 4}]
