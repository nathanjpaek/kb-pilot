import torch
import torch.nn as nn


class MyRelu(nn.Module):

    def __init__(self):
        super().__init__()
        self.myrelu1 = nn.ReLU()

    def forward(self, x):
        out1 = self.myrelu1(x)
        return out1


class FFNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(FFNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.lh = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.lr = nn.Linear(output_size, hidden_size)
        self.relu2 = MyRelu()

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.relu2(out1)
        out3 = self.lh(out2)
        out4 = self.relu2(out3)
        out5 = self.l2(out4)
        outr = out1 - self.lr(out5)
        out7 = self.relu2(outr)
        out8 = self.lh(out7)
        out9 = self.relu2(out8)
        out10 = self.l2(out9)
        return out10


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}]
