import torch


class MaskedLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y, Y, mask):
        diff = (y - Y) / 5.0
        return torch.mean(torch.square(diff[mask]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones(
        [4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
