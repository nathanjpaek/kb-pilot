import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class ModelRegressionGex2Atac(nn.Module):

    def __init__(self, dim_mod1, dim_mod2):
        super(ModelRegressionGex2Atac, self).__init__()
        self.input_ = nn.Linear(dim_mod1, 1024)
        self.fc = nn.Linear(1024, 256)
        self.fc1 = nn.Linear(256, 2048)
        self.dropout1 = nn.Dropout(p=0.298885630228993)
        self.dropout2 = nn.Dropout(p=0.11289717442776658)
        self.dropout3 = nn.Dropout(p=0.13523634924414762)
        self.output = nn.Linear(2048, dim_mod2)

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
