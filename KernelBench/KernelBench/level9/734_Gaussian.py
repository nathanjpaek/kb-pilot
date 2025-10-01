import torch


class Gaussian(torch.nn.Module):
    """Gaussian activation"""

    def forward(self, x):
        return torch.exp(-x * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
