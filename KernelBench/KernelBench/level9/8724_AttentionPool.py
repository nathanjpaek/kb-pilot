import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""

    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-07

    def forward(self, logits, decision):
        w = self.activ(self.transform(logits))
        detect = (decision * w).sum(self.pooldim) / (w.sum(self.pooldim) +
            self.eps)
        return detect


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 10])]


def get_init_inputs():
    return [[], {'inputdim': 4}]
