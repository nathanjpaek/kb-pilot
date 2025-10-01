import torch


class Add(torch.nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
