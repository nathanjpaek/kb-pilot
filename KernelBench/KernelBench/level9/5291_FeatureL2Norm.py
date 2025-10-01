import torch
import torch.nn as nn
import torch.nn


class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-06
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5
            ).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
