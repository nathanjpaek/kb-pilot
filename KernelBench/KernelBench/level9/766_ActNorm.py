import torch


class ActNorm(torch.nn.Module):

    def __init__(self, dim):
        super(type(self), self).__init__()
        self.dim = dim
        self.s = torch.nn.Parameter(torch.ones(1, dim))
        self.b = torch.nn.Parameter(torch.zeros(1, dim))
        return

    def forward(self, h):
        h = self.s * h + self.b
        logdet = self.dim * self.s.abs().log().sum()
        return h, logdet

    def reverse(self, h):
        h = (h - self.b) / self.s
        return h

    def latent(self, h):
        return self.forward(h)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
