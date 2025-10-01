import torch


class Normalize(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Normalize, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return torch.nn.functional.normalize(x, *self.args, **self.kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
