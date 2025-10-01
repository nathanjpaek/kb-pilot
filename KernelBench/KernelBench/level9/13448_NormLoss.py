import torch


class NormLoss(torch.nn.Module):
    """
    Norm penalty on function

    parameters:
        p - dimension of norm
    """

    def __init__(self, p):
        super(NormLoss, self).__init__()
        self.p = p

    def forward(self, beta):
        return torch.norm(beta, p=self.p)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 4}]
