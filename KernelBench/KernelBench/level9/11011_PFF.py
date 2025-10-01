import torch
import torch.nn as nn


class PFF(nn.Module):

    def __init__(self, model_dimension, width_mult=4):
        super().__init__()
        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.norm(self.linear2(self.dropout(self.relu(self.linear1(
            representations_batch)))) + representations_batch)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dimension': 4}]
