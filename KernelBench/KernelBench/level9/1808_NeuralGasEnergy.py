import torch


class NeuralGasEnergy(torch.nn.Module):

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, d):
        order = torch.argsort(d, dim=1)
        ranks = torch.argsort(order, dim=1)
        cost = torch.sum(self._nghood_fn(ranks, self.lm) * d)
        return cost, order

    def extra_repr(self):
        return f'lambda: {self.lm}'

    @staticmethod
    def _nghood_fn(rankings, lm):
        return torch.exp(-rankings / lm)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lm': 4}]
