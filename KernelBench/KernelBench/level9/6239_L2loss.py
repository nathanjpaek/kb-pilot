import torch


class L2loss(torch.nn.Module):

    def __init__(self):
        super(L2loss, self).__init__()

    def forward(self, y, yhat):
        loss = (y - yhat).pow(2).sum() / y.shape[0]
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
