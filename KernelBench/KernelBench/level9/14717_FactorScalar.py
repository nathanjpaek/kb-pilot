import torch
import torch.nn as nn


class FactorScalar(nn.Module):

    def __init__(self, initial_value=1.0, **kwargs):
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(initial_value))

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.forward(other)

    def __rmul__(self, other):
        return self.forward(other)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
