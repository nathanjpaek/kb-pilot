import torch
import torch.nn.init


class NormalizationLayer(torch.nn.Module):
    """Class for normalization layer."""

    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True
            ).expand_as(x)
        return features


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
