import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProductLoss(nn.Module):
    """This is the inner-product loss used in CFKG for optimization.
    """

    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        pos_score = torch.mul(anchor, positive).sum(dim=1)
        neg_score = torch.mul(anchor, negative).sum(dim=1)
        return (F.softplus(-pos_score) + F.softplus(neg_score)).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
