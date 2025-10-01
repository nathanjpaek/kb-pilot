import torch
from torch import nn


class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size, drop_p=0.2):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 102)
        self.dropout = nn.Dropout(p=drop_p)
        self.activation = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output_one = self.fc2(relu)
        output_final = self.activation(output_one)
        return output_final


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
