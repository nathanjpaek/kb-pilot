import torch
import torch.nn as nn
import torch.utils.model_zoo


class Charbonnier(nn.Module):

    def __init__(self):
        super(Charbonnier, self).__init__()
        self.eps = 1e-06

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
