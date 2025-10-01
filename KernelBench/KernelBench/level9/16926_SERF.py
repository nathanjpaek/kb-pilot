import torch
import torch.nn as nn


class SERF(nn.Module):

    def __init__(self, thresh=50):
        super().__init__()
        self.thresh = thresh
        None

    def forward(self, x):
        return self.serf_log1pexp(x)

    def serf(self, x):
        return x * torch.erf(torch.log(1 + torch.exp(x)))

    def serf_log1pexp(self, x):
        return x * torch.erf(torch.log1p(torch.exp(torch.clamp(x, max=self.
            thresh))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
