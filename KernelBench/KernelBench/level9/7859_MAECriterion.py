import torch


class MAECriterion(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
