import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.input_fc(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_fc(x))
        x = self.dropout(x)
        outputs = self.output_fc(x)
        return outputs, x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
