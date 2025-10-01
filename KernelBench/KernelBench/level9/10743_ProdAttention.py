import math
import torch
import torch.nn as nn
import torch.optim


class ProdAttention(nn.Module):

    def __init__(self, log_t=False):
        super(ProdAttention, self).__init__()
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None):
        pax = eh * dhx
        pax = torch.sum(pax, dim=2)
        if self.log_t:
            log_t = math.log(pax.size()[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax, dim=1)
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
