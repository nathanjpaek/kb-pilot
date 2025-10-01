import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 20)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(20, 20)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(20, 20)
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(20, n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4, 'n_output': 4}]
