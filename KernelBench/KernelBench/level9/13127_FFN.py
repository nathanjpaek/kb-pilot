import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """
    Feed-Forward Network
    """

    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = torch.nn.Linear(in_features=d_model, out_features=
            d_inner_hid)
        self.fc2 = torch.nn.Linear(in_features=d_inner_hid, out_features=
            d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        if self.dropout_rate:
            hidden = F.dropout(hidden, p=self.dropout_rate)
        out = self.fc2(hidden)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_inner_hid': 4, 'd_model': 4, 'dropout_rate': 0.5}]
