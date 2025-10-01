import math
import torch
import torchvision.transforms.functional as F
from torch import nn
import torch.nn.functional as F


class InteractionLayer(nn.Module):

    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        self.d_feature = d_feature
        self.det_tfm = nn.Linear(d_model, d_feature)
        self.rel_tfm = nn.Linear(d_model, d_feature)
        self.det_value_tfm = nn.Linear(d_model, d_feature)
        self.rel_norm = nn.LayerNorm(d_model)
        if dropout is not None:
            self.dropout = dropout
            self.det_dropout = nn.Dropout(dropout)
            self.rel_add_dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, det_in, rel_in):
        det_attn_in = self.det_tfm(det_in)
        rel_attn_in = self.rel_tfm(rel_in)
        det_value = self.det_value_tfm(det_in)
        scores = torch.matmul(det_attn_in.transpose(0, 1), rel_attn_in.
            permute(1, 2, 0)) / math.sqrt(self.d_feature)
        det_weight = F.softmax(scores.transpose(1, 2), dim=-1)
        if self.dropout is not None:
            det_weight = self.det_dropout(det_weight)
        rel_add = torch.matmul(det_weight, det_value.transpose(0, 1))
        rel_out = self.rel_add_dropout(rel_add) + rel_in.transpose(0, 1)
        rel_out = self.rel_norm(rel_out)
        return det_in, rel_out.transpose(0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_feature': 4}]
