import torch
import torch.nn.functional as F
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: 'int', hidden_size: 'int'=None, dropout:
        'float'=None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' in n:
                torch.nn.init.zeros_(p)
            elif 'fc' in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
