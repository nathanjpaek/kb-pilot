import torch
import torch.nn as nn


class ThumbInstanceNorm(nn.Module):

    def __init__(self, out_channels=None, affine=True):
        super(ThumbInstanceNorm, self).__init__()
        self.thumb_mean = None
        self.thumb_std = None
        self.collection = True
        if affine is True:
            self.weight = nn.Parameter(torch.ones(size=(1, out_channels, 1,
                1), requires_grad=True))
            self.bias = nn.Parameter(torch.zeros(size=(1, out_channels, 1, 
                1), requires_grad=True))

    def calc_mean_std(self, feat, eps=1e-05):
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x, thumb=None):
        if self.training:
            thumb_mean, thumb_std = self.calc_mean_std(thumb)
            x = (x - thumb_mean) / thumb_std * self.weight + self.bias
            thumb = (thumb - thumb_mean) / thumb_std * self.weight + self.bias
            return x, thumb
        else:
            if self.collection:
                thumb_mean, thumb_std = self.calc_mean_std(x)
                self.thumb_mean = thumb_mean
                self.thumb_std = thumb_std
            x = (x - self.thumb_mean
                ) / self.thumb_std * self.weight + self.bias
            return x


class ThumbAdaptiveInstanceNorm(ThumbInstanceNorm):

    def __init__(self):
        super(ThumbAdaptiveInstanceNorm, self).__init__(affine=False)

    def forward(self, content_feat, style_feat):
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        if self.collection is True:
            thumb_mean, thumb_std = self.calc_mean_std(content_feat)
            self.thumb_mean = thumb_mean
            self.thumb_std = thumb_std
        normalized_feat = (content_feat - self.thumb_mean.expand(size)
            ) / self.thumb_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(
            size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
