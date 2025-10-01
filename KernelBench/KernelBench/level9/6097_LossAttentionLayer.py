import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class LossAttentionLayer(nn.Module):

    def __init__(self):
        super(LossAttentionLayer, self).__init__()

    def forward(self, features, W_1, b_1):
        out_c = F.linear(features, W_1, b_1)
        out = out_c - out_c.max()
        out = out.exp()
        out = out.sum(1, keepdim=True)
        alpha = out / out.sum(0)
        alpha01 = features.size(0) * alpha.expand_as(features)
        context = torch.mul(features, alpha01)
        return context, out_c, torch.squeeze(alpha)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
