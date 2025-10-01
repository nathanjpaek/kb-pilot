import torch


class WeightedLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y, Y, w):
        diff = (y - Y) / 5.0
        return torch.mean(torch.square(diff) * w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
