import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        hidden2_size = int(input_size / 2)
        hidden1_size = int((input_size + hidden2_size) * 3 / 2)
        hidden3_size = int((output_size + hidden2_size) * 3 / 2)
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.hidden3 = nn.Linear(hidden2_size, hidden3_size)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.predict = nn.Linear(hidden3_size, output_size)
        nn.init.xavier_uniform_(self.predict.weight)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
