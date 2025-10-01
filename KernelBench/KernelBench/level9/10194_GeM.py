import torch
import torch.nn as nn


class GeM(nn.Module):

    def __init__(self, dim=1, p=0.0, eps=1e-06):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(()) * p, requires_grad=True)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-06):
        x_max = x.max(dim=-1, keepdim=False)[0]
        x_avg = x.mean(dim=-1, keepdim=False)
        w = torch.sigmoid(self.p)
        x = w * x_max + (1 - w) * x_avg
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.2f}'.format(self.p
            ) + ', ' + 'eps=' + str(self.eps) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
