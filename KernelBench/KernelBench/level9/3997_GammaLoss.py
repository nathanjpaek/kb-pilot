import torch
import torch.nn


class GammaLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y, y_hat):
        p = 2
        loss = -y * torch.pow(y_hat, 1 - p) / (1 - p) + torch.pow(y_hat, 2 - p
            ) / (2 - p)
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
