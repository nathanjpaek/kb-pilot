import torch


def tensor_min(input, dim, keepdim=False):
    if isinstance(dim, int):
        return torch.min(input, dim=dim, keepdim=keepdim)[0]
    else:
        if isinstance(dim, tuple):
            dim = list(dim)
        for d in dim:
            input = torch.min(input, dim=d, keepdim=keepdim)[0]
        return input


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


class TensorMin(StatModule, torch.nn.Module):

    def forward(self, input):
        return tensor_min(input, dim=self.dim, keepdim=self.keepdim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
