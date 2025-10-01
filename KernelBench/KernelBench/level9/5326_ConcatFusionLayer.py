import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusionLayer(nn.Module):

    def __init__(self, dim_model, voc_size, dout_p):
        super(ConcatFusionLayer, self).__init__()
        self.linear = nn.Linear(dim_model, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        return F.log_softmax(x, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4, 'voc_size': 4, 'dout_p': 0.5}]
