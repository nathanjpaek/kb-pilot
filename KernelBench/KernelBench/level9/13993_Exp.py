import torch


class Exp(torch.nn.Module):

    def forward(self, x):
        return (-0.5 * x ** 2).exp()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
