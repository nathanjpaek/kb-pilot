import torch
import torch.nn as nn


class GRU122(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRU122, self).__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=2 *
            hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=2 *
            hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=2 *
            hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=2 *
            hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=2 *
            hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=2 *
            hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        dh = h.repeat(1, 1, 2)
        out = (1 - z) * n + z * dh
        return torch.split(out, self.hidden_size, dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
