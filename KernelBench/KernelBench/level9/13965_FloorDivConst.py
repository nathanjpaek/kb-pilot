import torch


class FloorDivConst(torch.nn.Module):

    def __init__(self):
        super(FloorDivConst, self).__init__()

    def forward(self, x):
        return x // 2.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
