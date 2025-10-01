import torch
import torch.nn as nn
import torch.cuda
import torch.distributed


class LayerNormAVG(nn.Module):
    """
        Layer Normalization class inspired by Transformer normalization, but here we normalize to given average
        to preserve magnitue of USE
    """

    def __init__(self, features, desired_avg, eps=1e-06):
        super(LayerNormAVG, self).__init__()
        self.desiredAVG = desired_avg
        self.eps = eps
        self.size = features

    def forward(self, x):
        to_norm = torch.sqrt(self.desiredAVG * self.size / torch.sum(x ** 2))
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        ret = (x - mean) / (std + self.eps)
        return to_norm * ret


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4, 'desired_avg': 4}]
