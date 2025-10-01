import torch


def min_max_normalizer(x, detach=False):
    x_min = torch.min(x)
    x_max = torch.max(x)
    if detach:
        x_min = x_min.detach()
        x_max = x_max.detach()
    return (x - x_min) / (x_max - x_min)


class MinMaxNormalizer(torch.nn.Module):

    def __init__(self, detach=False):
        super().__init__()
        self.detach = detach

    def forward(self, x):
        return min_max_normalizer(x, detach=self.detach)

    def extra_repr(self):
        return c_f.extra_repr(self, ['detach'])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
