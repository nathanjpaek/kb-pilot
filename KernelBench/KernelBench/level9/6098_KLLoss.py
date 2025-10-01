import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


def kl_loss(x, y):
    x = F.softmax(x.detach(), dim=1)
    y = F.log_softmax(y, dim=1)
    return torch.mean(torch.sum(x * (torch.log(x) - y), dim=1))


class KLLoss(nn.Module):

    def forward(self, x, y):
        return kl_loss(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
