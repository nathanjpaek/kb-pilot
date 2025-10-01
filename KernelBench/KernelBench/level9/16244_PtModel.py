import torch
import torch.nn as nn


class PtModel(nn.Module):

    def __init__(self, n_features, n_labels, softmax=False, dropout=False):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, n_labels)
        self.dropout = nn.Dropout(0.5) if dropout else lambda x: x
        self.softmax = nn.Softmax() if softmax else lambda x: x

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'n_labels': 4}]
