import torch
import torch.nn as nn


class GateContextSelectionLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout):
        super(GateContextSelectionLayer, self).__init__()
        self.source = nn.Linear(dim_model, dim_model)
        self.context = nn.Linear(dim_model, dim_model)

    def forward(self, x_1, x_2, *args):
        update = torch.sigmoid(self.source(x_1) + self.context(x_2))
        out = (1 - update) * x_1 + update * x_2
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4, 'dim_ff': 4, 'prob_dropout': 0.5}]
