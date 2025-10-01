import torch
import torch.nn as nn
import torch.nn.functional as F


class TFBCELoss(nn.Module):

    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        relu_logits = F.relu(logits)
        neg_abs_logits = -torch.abs(logits)
        term1 = relu_logits - logits * targets
        term2 = torch.log1p(torch.exp(neg_abs_logits))
        loss = term1 + term2
        loss = loss.sum(dim=-1).mean(dim=-1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pos_weight': 4}]
