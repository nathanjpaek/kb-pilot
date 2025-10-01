import torch
import torch.nn as nn
import torch.utils.data


class Linear_2L(nn.Module):

    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_2L, self).__init__()
        self.n_hid = n_hid
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(self.n_hid, output_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        y = self.fc3(x)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'n_hid': 4}]
