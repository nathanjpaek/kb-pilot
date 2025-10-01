import torch
from torchvision.transforms import functional as F
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class Gated_Recurrent_Unit(nn.Module):

    def __init__(self, fea_size, dropout):
        super(Gated_Recurrent_Unit, self).__init__()
        self.wih = nn.Linear(fea_size, fea_size, bias=True)
        self.whh = nn.Linear(fea_size, fea_size, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'fea_size': 4, 'dropout': 0.5}]
