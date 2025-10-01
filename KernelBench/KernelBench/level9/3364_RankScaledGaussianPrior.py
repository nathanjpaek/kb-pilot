import torch


def rank_scaled_gaussian(distances, lambd):
    order = torch.argsort(distances, dim=1)
    ranks = torch.argsort(order, dim=1)
    return torch.exp(-torch.exp(-ranks / lambd) * distances)


class RankScaledGaussianPrior(torch.nn.Module):

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, distances):
        return rank_scaled_gaussian(distances, self.lambd)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lambd': 4}]
