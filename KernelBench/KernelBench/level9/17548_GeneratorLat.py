import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F


class GeneratorLat(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, tgt_lat_classes):
        super(GeneratorLat, self).__init__()
        self.proj = nn.Linear(d_model, tgt_lat_classes)

    def forward(self, x):
        lat_pred = F.softmax(self.proj(x), dim=-1)
        lat_pred = lat_pred[:, -1, :]
        lat_pred = torch.squeeze(lat_pred)
        return lat_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'tgt_lat_classes': 4}]
