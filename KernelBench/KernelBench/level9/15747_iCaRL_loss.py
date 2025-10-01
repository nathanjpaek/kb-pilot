import torch
import torch.nn as nn


class iCaRL_loss(nn.Module):

    def __init__(self):
        super(iCaRL_loss, self).__init__()

    def forward(self, logist, target):
        eps = 1e-06
        logist = logist.double()
        target = target.double()
        p0 = torch.mul(target, torch.log(logist + eps))
        p1 = torch.mul(1 - target, torch.log(1 - logist + eps))
        loss = -torch.add(p0, p1)
        loss = torch.sum(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
