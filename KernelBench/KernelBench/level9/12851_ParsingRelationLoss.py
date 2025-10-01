import torch
import torch.nn.modules
import torch.nn as nn


class ParsingRelationLoss(nn.Module):

    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        _n, _c, h, _w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
