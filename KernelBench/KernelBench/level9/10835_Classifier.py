import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, inputs, hidden_units):
        super().__init__()
        self.hidden = nn.Linear(inputs, hidden_units)
        self.output = nn.Linear(hidden_units, 102)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputs': 4, 'hidden_units': 4}]
