import torch


class StatModule(torch.nn.Module):

    def __init__(self, dim, keepdim=False):
        if isinstance(dim, list):
            dim = tuple(dim)
        if isinstance(dim, int):
            dim = dim,
        assert isinstance(dim, tuple)
        self.dim = dim
        self.keepdim = keepdim
        super().__init__()


class TensorMean(StatModule):

    def forward(self, input):
        return torch.mean(input, dim=self.dim, keepdim=self.keepdim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
