import torch


class UnbalancedWeight(torch.nn.Module):

    def __init__(self, ε, ρ):
        super(UnbalancedWeight, self).__init__()
        self.ε, self.ρ = ε, ρ

    def forward(self, x):
        return (self.ρ + self.ε / 2) * x

    def backward(self, g):
        return (self.ρ + self.ε) * g


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ε': 4, 'ρ': 4}]
