import torch


class pair_norm(torch.nn.Module):

    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-06 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
