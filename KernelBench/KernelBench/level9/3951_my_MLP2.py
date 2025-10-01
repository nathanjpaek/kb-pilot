import torch
import torch.nn as nn
import torch.nn.functional as F


class my_MLP2(nn.Module):

    def __init__(self, input_dim, output_dim, softmax_type='vanilla'):
        super().__init__()
        self.input = nn.Linear(input_dim, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.softmax_type = softmax_type
        self.hyp = nn.Linear(128, 1)

    def forward(self, parameters):
        l_1 = self.input(parameters)
        l_2 = F.relu(self.hidden1(l_1))
        l_3 = F.relu(self.hidden2(l_2))
        l_4 = F.relu(self.hidden3(l_3))
        if self.softmax_type == 'vanilla':
            w_pred = self.softmax(self.output(l_4))
        elif self.softmax_type == 'radius_one':
            w_pred = self.softmax(self.output(l_4)) * 2 - 1
        else:
            w_pred = self.output(l_4)
        hyp = torch.sigmoid(self.hyp(l_3)) * 5 + 1
        return w_pred, hyp


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
