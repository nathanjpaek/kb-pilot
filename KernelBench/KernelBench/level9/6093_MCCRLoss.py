import torch
from torch import nn


class MCCRLoss(nn.Module):
    """Maximum Correntropy Criterion Induced Losses for Regression(MCCR) Loss"""

    def __init__(self, sigma=1.0):
        super().__init__()
        assert sigma > 0
        self.sigma2 = sigma ** 2

    def forward(self, _input: 'torch.Tensor', _target: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Implement maximum correntropy criterion for regression

        loss(y, t) = sigma^2 * (1.0 - exp(-(y-t)^2/sigma^2))

        where sigma > 0 (parameter)

        Reference:
            * Feng, Yunlong, et al.
                "Learning with the maximum correntropy criterion
                    induced losses for regression."
                J. Mach. Learn. Res. 16.1 (2015): 993-1034.
        """
        return torch.mean(self.sigma2 * (1 - torch.exp(-(_input - _target) **
            2 / self.sigma2)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
