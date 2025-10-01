import torch
import torch.nn


class LogCoshLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = torch.abs(y_t - y_prime_t)
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-16)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
