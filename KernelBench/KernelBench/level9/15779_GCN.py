from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):

    def __init__(self, cfg):
        super(GCN, self).__init__()
        self.num_layers = cfg.num_layers
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.dropout = cfg.dropout
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_size, self.
            hidden_size) for i in range(self.num_layers - 1)])
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, adj):
        L = x.size(1)
        AxW = self.fc1(torch.bmm(adj, x)) + self.fc1(x)
        AxW = AxW / L
        AxW = F.leaky_relu(AxW)
        AxW = self.dropout(AxW)
        for fc in self.fcs:
            AxW = fc(torch.bmm(adj, AxW)) + fc(AxW)
            AxW = AxW / L
            AxW = F.leaky_relu(AxW)
            AxW = self.dropout(AxW)
        return AxW


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(num_layers=1, input_size=4,
        hidden_size=4, dropout=0.5)}]
