import torch
from torch import nn
from torch import einsum


class SelfDisLoss(nn.Module):

    def __init__(self):
        super(SelfDisLoss, self).__init__()

    def forward(self, feat, mean_feat):
        sim = einsum('nc,nc->n', [feat, mean_feat])
        dis = torch.sqrt(2.0 * (1 - sim))
        loss = torch.mean(dis)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
