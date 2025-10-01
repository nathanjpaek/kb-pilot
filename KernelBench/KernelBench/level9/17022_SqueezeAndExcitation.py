import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class SqueezeAndExcitation(nn.Module):

    def __init__(self, planes, squeeze):
        super(SqueezeAndExcitation, self).__init__()
        self.squeeze = nn.Linear(planes, squeeze)
        self.expand = nn.Linear(squeeze, planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        out = self.squeeze(out)
        out = self.relu(out)
        out = self.expand(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'planes': 4, 'squeeze': 4}]
