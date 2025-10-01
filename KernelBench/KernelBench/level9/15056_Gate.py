import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):

    def __init__(self, hidden_size):
        super(Gate, self).__init__()
        self.hidden_size = hidden_size
        self.wrx = nn.Linear(hidden_size, hidden_size)
        self.wrh = nn.Linear(hidden_size, hidden_size)
        self.wix = nn.Linear(hidden_size, hidden_size)
        self.wih = nn.Linear(hidden_size, hidden_size)
        self.wnx = nn.Linear(hidden_size, hidden_size)
        self.wnh = nn.Linear(hidden_size, hidden_size)

    def forward(self, title, pg):
        r_gate = F.sigmoid(self.wrx(title) + self.wrh(pg))
        i_gate = F.sigmoid(self.wix(title) + self.wih(pg))
        n_gate = F.tanh(self.wnx(title) + torch.mul(r_gate, self.wnh(pg)))
        result = torch.mul(i_gate, pg) + torch.mul(torch.add(-i_gate, 1),
            n_gate)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
