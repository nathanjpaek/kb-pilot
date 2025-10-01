import torch
import torch.nn as nn


class HDRLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out_img, ref_img):
        return torch.mean((out_img - ref_img) ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
