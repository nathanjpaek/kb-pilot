import torch
from torch import nn


class DNNModel(nn.Module):

    def __init__(self, dropout=0.2):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(4, 4)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(4, 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.relu_last = nn.ReLU()
        self.fc_last = nn.Linear(4, 3)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc_last(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
