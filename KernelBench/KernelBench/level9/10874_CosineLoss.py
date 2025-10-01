import torch
import torch.nn as nn


class CosineLoss(nn.Module):
    cos = nn.CosineSimilarity(dim=2, eps=1e-06)

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask.unsqueeze(2))
        return (1.0 - self.cos(pred, target)).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
