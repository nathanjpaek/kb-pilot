import torch


class PairwiseDistance(torch.nn.Module):

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x, y):
        x_ = x.repeat([1] + list(y.shape[1:])).reshape(*y.shape, -1)
        y_ = y.repeat([1] + list(x.shape[1:])).reshape(*x.shape, -1).transpose(
            -1, -2)
        return x_.sub(y_).abs().pow(self.p)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
