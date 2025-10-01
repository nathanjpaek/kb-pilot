import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class ModelRegressionAtac2Gex(nn.Module):

    def __init__(self, dim_mod1, dim_mod2):
        super(ModelRegressionAtac2Gex, self).__init__()
        self.input_ = nn.Linear(dim_mod1, 2048)
        self.fc = nn.Linear(2048, 2048)
        self.fc1 = nn.Linear(2048, 512)
        self.dropout1 = nn.Dropout(p=0.2649138776004753)
        self.dropout2 = nn.Dropout(p=0.1769628308148758)
        self.dropout3 = nn.Dropout(p=0.2516791883012817)
        self.output = nn.Linear(512, dim_mod2)

    def forward(self, x):
        x = F.gelu(self.input_(x))
        x = self.dropout1(x)
        x = F.gelu(self.fc(x))
        x = self.dropout2(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout3(x)
        x = F.gelu(self.output(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_mod1': 4, 'dim_mod2': 4}]
