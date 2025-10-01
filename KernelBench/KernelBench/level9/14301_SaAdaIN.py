import torch
import torch.nn as nn
import torch.optim
import torch.utils.data


def calc_mean_std(feat, eps=1e-05):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class SaAdaIN(nn.Module):
    """
    Sandwich Adaptive-Instance-Normalization.
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.shared_weight = nn.Parameter(torch.Tensor(num_features),
            requires_grad=True)
        self.shared_bias = nn.Parameter(torch.Tensor(num_features),
            requires_grad=True)
        nn.init.ones_(self.shared_weight)
        nn.init.zeros_(self.shared_bias)

    def forward(self, content_feat, style_feat):
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)
            ) / content_std.expand(size)
        shared_affine_feat = normalized_feat * self.shared_weight.view(1,
            self.num_features, 1, 1).expand(size) + self.shared_bias.view(1,
            self.num_features, 1, 1).expand(size)
        output = shared_affine_feat * style_std.expand(size
            ) + style_mean.expand(size)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
