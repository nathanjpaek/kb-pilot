import torch


class AbsLoss(torch.nn.Module):
    """
    The mean absolute value.
    """

    def forward(self, x):
        """"""
        return torch.mean(torch.abs(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
