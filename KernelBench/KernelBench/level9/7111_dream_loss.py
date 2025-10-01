import torch


class dream_loss(torch.nn.Module):

    def __init__(self):
        super(dream_loss, self).__init__()

    def forward(self, yhat, y):
        diff = torch.sum(yhat - y)
        return diff


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
