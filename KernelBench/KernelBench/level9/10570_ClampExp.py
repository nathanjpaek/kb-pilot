import torch
import torch.utils.data


class ClampExp(torch.nn.Module):
    """
    Nonlinearity min(exp(lam * x), 1)
    """

    def __init__(self):
        """
        Constructor
        :param lam: Lambda parameter
        """
        super(ClampExp, self).__init__()

    def forward(self, x):
        one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return torch.min(torch.exp(x), one)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
