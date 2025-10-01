import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


class forfilter(nn.Module):

    def __init__(self, inplanes):
        super(forfilter, self).__init__()
        self.forfilter1 = nn.Conv2d(1, 1, (7, 1), 1, (0, 0), bias=False)
        self.inplanes = inplanes

    def forward(self, x):
        out = self.forfilter1(F.pad(torch.unsqueeze(x[:, 0, :, :], 1), pad=
            (0, 0, 3, 3), mode='replicate'))
        for i in range(1, self.inplanes):
            out = torch.cat((out, self.forfilter1(F.pad(torch.unsqueeze(x[:,
                i, :, :], 1), pad=(0, 0, 3, 3), mode='replicate'))), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4}]
