import torch
import torch.nn as nn


class JS_Divergence(nn.Module):

    def __init__(self):
        super().__init__()
        self.engine = nn.KLDivLoss()

    def forward(self, x, y):
        return self.engine(x, y) + self.engine(y, x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
