import torch
import torch.fft
import torch.nn


class MultiplyLearned(torch.nn.Module):

    def __init__(self, omega_0: 'float'):
        """
        out = omega_0 * x, with a learned omega_0
        """
        super().__init__()
        self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
        with torch.no_grad():
            self.omega_0.fill_(omega_0)

    def forward(self, x):
        return 100 * self.omega_0 * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'omega_0': 4}]
