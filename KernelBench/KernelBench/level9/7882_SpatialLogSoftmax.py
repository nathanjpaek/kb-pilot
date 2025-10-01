import torch
import torch.utils.data
import torch.random
import torch.nn.functional as F


def logprob_to_keypoints(prob, length):
    ruler = torch.log(torch.linspace(0, 1, length, device=prob.device)
        ).type_as(prob).expand(1, 1, -1)
    return torch.sum(torch.exp(prob + ruler), dim=2, keepdim=True).squeeze(2)


def marginal_logsoftmax(heatmap, dim):
    marginal = torch.mean(heatmap, dim=dim)
    sm = F.log_softmax(marginal, dim=2)
    return sm


def spacial_logsoftmax(heatmap, probs=False):
    height, width = heatmap.size(2), heatmap.size(3)
    hp, wp = marginal_logsoftmax(heatmap, dim=3), marginal_logsoftmax(heatmap,
        dim=2)
    hk, wk = logprob_to_keypoints(hp, height), logprob_to_keypoints(wp, width)
    if probs:
        return torch.stack((hk, wk), dim=2), (torch.exp(hp), torch.exp(wp))
    else:
        return torch.stack((hk, wk), dim=2)


class SpatialLogSoftmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, heatmap, probs=False):
        return spacial_logsoftmax(heatmap, probs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
