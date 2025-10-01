import torch
import torch.nn as nn


def project_simplex(x):
    """
    Project an arbitary vector onto the simplex.
    See [Wang & Carreira-Perpin 2013] for a description and references.
    """
    n = x.size()[0]
    mu = torch.sort(x, 0, descending=True)[0]
    sm = 0
    for j in xrange(1, n + 1):
        sm += mu[j - 1]
        t = mu[j - 1] - 1.0 / j * (sm - 1)
        if t > 0:
            row = j
            sm_row = sm
    theta = 1.0 / row * (sm_row - 1)
    y = x - theta
    return torch.clamp(y, min=0.0)


class MixtureSoftmax(nn.Module):
    """
    """

    def __init__(self, in_features):
        super(MixtureSoftmax, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_weights()

    def reset_weights(self):
        self.weight.data.normal_(10, 1.0)
        self.weight.data /= self.weight.data.sum()

    def forward(self, input):
        return torch.sum(input * self.weight[None, :, None], dim=1)

    def project_parameters(self):
        for p in self.parameters():
            p.data = project_simplex(p.data)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
