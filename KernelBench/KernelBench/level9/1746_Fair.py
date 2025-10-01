import torch
import torch.nn as nn
import torch.utils.model_zoo


class Fair(nn.Module):

    def __init__(self):
        super(Fair, self).__init__()
        self.c = 1.0

    def forward(self, X, Y):
        r = torch.add(X, -Y)
        ra = torch.abs(r)
        error = self.c ** 2 * (ra / self.c - torch.log(1 + ra / self.c))
        loss = torch.sum(error)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
