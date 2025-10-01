import torch
import torch.nn.functional as F


class GeometricMean(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        log_x = torch.log(F.relu(x))
        return torch.exp(torch.mean(log_x, dim=self.dim))


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
