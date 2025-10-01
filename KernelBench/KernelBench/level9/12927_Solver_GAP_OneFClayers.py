import torch
import torch.nn as nn
import torch.nn.functional as F


class Solver_GAP_OneFClayers(nn.Module):
    """ GAP + fc1 """

    def __init__(self, input_nc, input_width, input_height, dropout_prob=
        0.0, reduction_rate=2, **kwargs):
        super(Solver_GAP_OneFClayers, self).__init__()
        self.dropout_prob = dropout_prob
        self.reduction_rate = reduction_rate
        self.fc1 = nn.Linear(input_nc, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.mean(dim=-1).mean(dim=-1).squeeze()
        x = F.dropout(x, training=self.training, p=self.dropout_prob)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_nc': 4, 'input_width': 4, 'input_height': 4}]
