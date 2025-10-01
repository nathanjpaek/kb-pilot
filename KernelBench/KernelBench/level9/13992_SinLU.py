import torch


class SinLU(torch.nn.Module):

    def __init__(self):
        super(SinLU, self).__init__()
        self.thr = torch.nn.Threshold(0, 0)

    def forward(self, x):
        return self.thr(x) - self.thr(-x).sin()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
