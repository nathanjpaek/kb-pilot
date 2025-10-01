import torch
from torch.nn.modules.loss import _Loss


class TotalVariations(_Loss):

    def forward(self, img1):
        return torch.sum(torch.abs(img1[:, :, :-1] - img1[:, :, 1:])
            ) + torch.sum(torch.abs(img1[:, :-1, :] - img1[:, 1:, :]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
