import torch
import numpy as np
import torch.nn as nn


class DCCWeightedELoss(nn.Module):

    def __init__(self, size_average=True):
        super(DCCWeightedELoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, outputs, weights):
        out = (inputs - outputs).view(len(inputs), -1)
        out = torch.sum(weights * torch.norm(out, p=2, dim=1) ** 2)
        assert np.isfinite(out.data.cpu().numpy()).all(), 'Nan found in data'
        if self.size_average:
            out = out / inputs.nelement()
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
