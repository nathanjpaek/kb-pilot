import torch
import torch.optim
import torch.jit
import torch.nn as nn


class OptimizedMLP(nn.Module):

    def __init__(self, num_in_features: 'int', num_out_features: 'int'):
        super(OptimizedMLP, self).__init__()
        self.act = nn.ELU()
        self.l_in = nn.Linear(in_features=num_in_features, out_features=107)
        self.l1 = nn.Linear(in_features=107, out_features=179)
        self.l2 = nn.Linear(in_features=179, out_features=179)
        self.l3 = nn.Linear(in_features=179, out_features=184)
        self.l4 = nn.Linear(in_features=184, out_features=115)
        self.l_out = nn.Linear(in_features=115, out_features=num_out_features)
        torch.nn.init.xavier_normal_(self.l_in.weight)
        torch.nn.init.zeros_(self.l_in.bias)
        torch.nn.init.xavier_normal_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.xavier_normal_(self.l2.weight)
        torch.nn.init.zeros_(self.l2.bias)
        torch.nn.init.xavier_normal_(self.l3.weight)
        torch.nn.init.zeros_(self.l3.bias)
        torch.nn.init.xavier_normal_(self.l4.weight)
        torch.nn.init.zeros_(self.l4.bias)
        torch.nn.init.xavier_normal_(self.l_out.weight)
        torch.nn.init.zeros_(self.l_out.bias)

    def forward(self, x):
        x = self.act(self.l_in(x))
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.act(self.l4(x))
        x = self.l_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in_features': 4, 'num_out_features': 4}]
