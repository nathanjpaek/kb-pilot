import torch
import torch.nn as nn
import torch.nn.parallel


class CosineAngularLoss(nn.Module):

    def __init__(self):
        super(CosineAngularLoss, self).__init__()

    def forward(self, preds, truths):
        preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
        truths_norm = torch.nn.functional.normalize(truths, p=2, dim=1)
        loss = torch.mean(-torch.sum(preds_norm * truths_norm, dim=1))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
