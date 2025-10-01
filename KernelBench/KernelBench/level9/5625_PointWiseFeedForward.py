import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_affine, fc_dorpout=0.2):
        super().__init__()
        self.d_model = d_model
        self.d_affine = d_affine
        self.linear_1 = nn.Linear(self.d_model, self.d_affine)
        self.linear_2 = nn.Linear(self.d_affine, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_1 = nn.Dropout(fc_dorpout)
        self.dropout_2 = nn.Dropout(fc_dorpout)
        self.selu = nn.SELU()

    def forward(self, x):
        output = self.dropout_1(self.selu(self.linear_1(self.layer_norm(x))))
        output = self.dropout_2(self.linear_2(output))
        return output + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_affine': 4}]
