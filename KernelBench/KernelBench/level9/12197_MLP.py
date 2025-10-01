import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=500,
        weight_decay=0.0):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.Dropout = nn.Dropout(p=0.5)
        self.h2o = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.loss = nn.functional.cross_entropy
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.h2o(torch.relu(self.Dropout(self.i2h(x))))

    def predict(self, x):
        return self.forward(x).argmax(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
