import torch
import torch.nn as nn


def clip_data(data, max_norm):
    norms = torch.norm(data.reshape(data.shape[0], -1), dim=-1)
    scale = (max_norm / norms).clamp(max=1.0)
    data *= scale.reshape(-1, 1, 1, 1)
    return data


class ClipLayer(nn.Module):

    def __init__(self, max_norm):
        super(ClipLayer, self).__init__()
        self.max_norm = max_norm

    def forward(self, x):
        return clip_data(x, self.max_norm)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_norm': 4}]
