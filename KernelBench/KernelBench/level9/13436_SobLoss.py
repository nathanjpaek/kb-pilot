import torch


class SobLoss(torch.nn.Module):
    """
    Sobolev norm penalty on function
    (sum |x_{i} - x{i+1}|^p)^{1/p}

    parameters:
        p - dimension of norm
    """

    def __init__(self, p):
        super(SobLoss, self).__init__()
        self.p = p

    def forward(self, beta):
        hdiff = beta[1:] - beta[:-1]
        return torch.norm(hdiff, p=self.p)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 4}]
