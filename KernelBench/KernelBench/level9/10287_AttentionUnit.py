import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class AttentionUnit(nn.Module):

    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim
        self.sEmbed = nn.Linear(sDim, attDim)
        self.xEmbed = nn.Linear(xDim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

    def init_weights(self):
        init.normal_(self.sEmbed.weight, std=0.01)
        init.constant_(self.sEmbed.bias, 0)
        init.normal_(self.xEmbed.weight, std=0.01)
        init.constant_(self.xEmbed.bias, 0)
        init.normal_(self.wEmbed.weight, std=0.01)
        init.constant_(self.wEmbed.bias, 0)

    def forward(self, x, sPrev):
        batch_size, T, _ = x.size()
        x = x.view(-1, self.xDim)
        xProj = self.xEmbed(x)
        xProj = xProj.view(batch_size, T, -1)
        sPrev = sPrev.squeeze(0)
        sProj = self.sEmbed(sPrev)
        sProj = torch.unsqueeze(sProj, 1)
        sProj = sProj.expand(batch_size, T, self.attDim)
        sumTanh = torch.tanh(sProj + xProj)
        sumTanh = sumTanh.view(-1, self.attDim)
        vProj = self.wEmbed(sumTanh)
        vProj = vProj.view(batch_size, T)
        alpha = F.softmax(vProj, dim=1)
        return alpha


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'sDim': 4, 'xDim': 4, 'attDim': 4}]
