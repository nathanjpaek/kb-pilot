import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class FeatureCorrelation(nn.Module):

    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feat_a, feat_b):
        bs, c, h, w = feat_a.size()
        feat_a = feat_a.transpose(2, 3).contiguous().view(bs, c, h * w)
        feat_b = feat_b.view(bs, c, h * w).transpose(1, 2)
        feat_mul = torch.bmm(feat_b, feat_a)
        correlate_tensor = feat_mul.view(bs, h, w, h * w).transpose(2, 3
            ).transpose(1, 2)
        return correlate_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
