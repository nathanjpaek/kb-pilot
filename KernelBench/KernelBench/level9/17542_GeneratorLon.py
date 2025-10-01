import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F


class GeneratorLon(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, tgt_lon_classes):
        super(GeneratorLon, self).__init__()
        self.proj = nn.Linear(d_model, 2, tgt_lon_classes)

    def forward(self, x):
        lon_pred = F.softmax(self.proj(x), dim=-1)
        lon_pred = lon_pred[:, -1, :]
        lon_pred = torch.squeeze(lon_pred)
        return lon_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'tgt_lon_classes': 4}]
