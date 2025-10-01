import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        hlayer1 = int(input_size * 10)
        hlayer2 = int(input_size * 10 / 2)
        self.fc1 = nn.Linear(input_size, hlayer1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hlayer1, hlayer2)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(hlayer2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        a2 = self.fc2(h1)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
