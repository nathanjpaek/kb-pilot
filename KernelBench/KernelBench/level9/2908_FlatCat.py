import torch


class FlatCat(torch.nn.Module):

    def __init__(self):
        super(FlatCat, self).__init__()

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, 1, 1)
        y = y.view(y.shape[0], -1, 1, 1)
        return torch.cat([x, y], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
