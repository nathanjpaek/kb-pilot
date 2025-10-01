import torch
import torch.nn as nn
import torch.utils.data


def rmseOnFeatures(feature_difference):
    gt = torch.zeros_like(feature_difference)
    return torch.nn.functional.mse_loss(feature_difference, gt,
        size_average=False)


class RMSEFeaturesLoss(nn.Module):

    def __init__(self):
        super(RMSEFeaturesLoss, self).__init__()

    def forward(self, feature_difference):
        return rmseOnFeatures(feature_difference)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
