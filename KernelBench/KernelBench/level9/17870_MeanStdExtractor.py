import torch
from torch import nn


class MeanStdExtractor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_maps_batch):
        feature_maps_batch = feature_maps_batch.view(*feature_maps_batch.
            shape[:2], -1)
        feature_means_batch = feature_maps_batch.mean(dim=-1)
        feature_stds_batch = feature_maps_batch.std(dim=-1)
        return torch.cat((feature_means_batch, feature_stds_batch), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
