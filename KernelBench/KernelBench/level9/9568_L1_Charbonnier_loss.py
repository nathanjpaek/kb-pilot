import torch
import torch.nn as nn


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss loss function where the epsilon has been taken as 1e-3 from the paper"""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 0.001

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.sum(error)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
