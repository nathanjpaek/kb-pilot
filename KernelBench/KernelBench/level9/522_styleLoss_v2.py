import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-05):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class GramMatrix(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h * w)
        G = torch.bmm(f, f.transpose(1, 2))
        return G.div_(c * h * w)


class styleLoss_v2(nn.Module):

    def forward(self, input, target):
        _ib, _ic, _ih, _iw = input.size()
        mean_x, var_x = calc_mean_std(input)
        iCov = GramMatrix()(input)
        mean_y, var_y = calc_mean_std(target)
        tCov = GramMatrix()(target)
        loss = nn.MSELoss(size_average=True)(mean_x, mean_y) + nn.MSELoss(
            size_average=True)(var_x, var_y) + nn.MSELoss(size_average=True)(
            iCov, tCov)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
