import torch


class CosModule(torch.nn.Module):

    def __init__(self):
        super(CosModule, self).__init__()

    def forward(self, x):
        return torch.cos(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
