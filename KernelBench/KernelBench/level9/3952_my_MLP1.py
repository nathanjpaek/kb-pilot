import torch
import torch.nn as nn


class my_MLP1(nn.Module):

    def __init__(self, input_dim, npdf, h1_dim, h2_dim, norm_type='softmax'):
        super().__init__()
        self.input = nn.Linear(input_dim, h1_dim)
        self.hidden = nn.Linear(h1_dim, h2_dim)
        self.output = nn.Linear(h2_dim, npdf)
        self.hyp = nn.Linear(h2_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = torch.sigmoid
        self.norm_type = norm_type

    def forward(self, inputs):
        l_1 = self.sigmoid(self.input(inputs))
        l_2 = self.sigmoid(self.hidden(l_1))
        w_un = self.output(l_2)
        hyp = self.sigmoid(self.hyp(l_2))
        if self.norm_type == 'softmax':
            w_pred = self.softmax(w_un)
        elif self.norm_type == 'normalize':
            self.sigmoid(w_un)
            w_pred = (w_un / w_un.sum(axis=0)).sum(axis=0)
        else:
            w_pred = torch.abs(self.output(w_un))
        return w_pred, hyp


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'npdf': 4, 'h1_dim': 4, 'h2_dim': 4}]
