import torch
import torch.cuda
import torch.nn as nn


class ExpLayer(nn.Module):

    def __init__(self, vMF_kappa):
        super(ExpLayer, self).__init__()
        self.vMF_kappa = nn.Parameter(torch.Tensor([vMF_kappa]))

    def forward(self, x, binary=False):
        if binary:
            x = torch.exp(self.vMF_kappa * x) * (x > 0.55).type(torch.
                FloatTensor)
        else:
            x = torch.exp(self.vMF_kappa * x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'vMF_kappa': 4}]
