import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data


class WeightedBCELoss2d(nn.Module):

    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        logits = logits.view(-1)
        gt = labels.view(-1)
        loss = logits.clamp(min=0) - logits * gt + torch.log(1 + torch.exp(
            -logits.abs()))
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
