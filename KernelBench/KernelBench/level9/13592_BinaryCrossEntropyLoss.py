from torch.nn import Module
import torch


class BinaryCrossEntropyLoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, groundtruth, distr_params, mask):
        groundtruth = (groundtruth - groundtruth.min()) / (groundtruth.max(
            ) - groundtruth.min())
        loss = mask * (groundtruth * distr_params.clamp(min=1e-07).log() + 
            (1 - groundtruth) * (1 - distr_params).clamp(min=1e-07).log())
        loss = loss.flatten(2).sum(dim=2)
        return loss.flatten()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
