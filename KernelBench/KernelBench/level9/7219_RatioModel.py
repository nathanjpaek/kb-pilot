import torch
import torch.nn.functional as F


class RatioModel(torch.nn.Module):

    def __init__(self, D_in, hidden_unit_num):
        super().__init__()
        None
        self.l1 = torch.nn.Linear(D_in, hidden_unit_num)
        self.l2 = torch.nn.Linear(hidden_unit_num, hidden_unit_num)
        self.l3 = torch.nn.Linear(hidden_unit_num, 1)

    def forward(self, X):
        return F.softplus(self.l3(torch.tanh(self.l2(torch.tanh(self.l1(X))))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'hidden_unit_num': 4}]
