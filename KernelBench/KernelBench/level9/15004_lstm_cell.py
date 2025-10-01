import torch
import torch.nn as nn


class lstm_cell(nn.Module):

    def __init__(self, input_num, hidden_num):
        super(lstm_cell, self).__init__()
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.Wxi = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whi = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxf = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whf = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxc = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Whc = nn.Linear(self.hidden_num, self.hidden_num, bias=False)
        self.Wxo = nn.Linear(self.input_num, self.hidden_num, bias=True)
        self.Who = nn.Linear(self.hidden_num, self.hidden_num, bias=False)

    def forward(self, xt, ht_1, ct_1):
        it = torch.sigmoid(self.Wxi(xt) + self.Whi(ht_1))
        ft = torch.sigmoid(self.Wxf(xt) + self.Whf(ht_1))
        ot = torch.sigmoid(self.Wxo(xt) + self.Who(ht_1))
        ct = ft * ct_1 + it * torch.tanh(self.Wxc(xt) + self.Whc(ht_1))
        ht = ot * torch.tanh(ct)
        return ht, ct


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_num': 4, 'hidden_num': 4}]
