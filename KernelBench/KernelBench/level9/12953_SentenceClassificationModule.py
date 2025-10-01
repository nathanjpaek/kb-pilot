from torch.nn import Module
import torch
from torch.nn import functional as F
import torch.onnx


class SentenceClassificationModule(Module):

    def __init__(self, input_dimensions: 'int', hidden_dimensions: 'int',
        dropout: 'float'=0.3):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_dimensions, hidden_dimensions)
        self.layer_2 = torch.nn.Linear(hidden_dimensions, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.norm = torch.nn.LayerNorm(hidden_dimensions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dimensions': 4, 'hidden_dimensions': 4}]
