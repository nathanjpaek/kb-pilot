import torch
import torch.utils.data
import torch.random
import torch.nn.functional as F


def marginal_softmax(heatmap, dim):
    marginal = torch.mean(heatmap, dim=dim)
    sm = F.softmax(marginal, dim=2)
    return sm


def prob_to_keypoints(prob, length):
    ruler = torch.linspace(0, 1, length).type_as(prob).expand(1, 1, -1)
    return torch.sum(prob * ruler, dim=2, keepdim=True).squeeze(2)


def spacial_softmax(heatmap, probs=False):
    height, width = heatmap.size(2), heatmap.size(3)
    hp, wp = marginal_softmax(heatmap, dim=3), marginal_softmax(heatmap, dim=2)
    hk, wk = prob_to_keypoints(hp, height), prob_to_keypoints(wp, width)
    if probs:
        return torch.stack((hk, wk), dim=2), (hp, wp)
    else:
        return torch.stack((hk, wk), dim=2)


class SpatialSoftmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, heatmap, probs=False):
        return spacial_softmax(heatmap, probs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
