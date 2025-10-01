import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class ModelRegressionGex2Adt(nn.Module):

    def __init__(self, dim_mod1, dim_mod2):
        super(ModelRegressionGex2Adt, self).__init__()
        self.input_ = nn.Linear(dim_mod1, 512)
        self.dropout1 = nn.Dropout(p=0.20335661386636347)
        self.dropout2 = nn.Dropout(p=0.15395289261127876)
        self.dropout3 = nn.Dropout(p=0.16902655078832815)
        self.fc = nn.Linear(512, 512)
        self.fc1 = nn.Linear(512, 2048)
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
