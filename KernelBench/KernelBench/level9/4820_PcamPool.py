import torch
import torch.nn as nn


class PcamPool(nn.Module):

    def __init__(self):
        super(PcamPool, self).__init__()

    def forward(self, feat_map, logit_map):
        assert logit_map is not None
        prob_map = torch.sigmoid(logit_map)
        weight_map = prob_map / prob_map.sum(dim=2, keepdim=True).sum(dim=3,
            keepdim=True)
        feat = (feat_map * weight_map).sum(dim=2, keepdim=True).sum(dim=3,
            keepdim=True)
        return feat


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
