import torch
from torch import nn


class AttentionLoss(nn.Module):

    def __init__(self, beta=4, gamma=0.5):
        super(AttentionLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = torch.sum(gt)
        num_neg = torch.sum(1 - gt)
        alpha = num_neg / (num_pos + num_neg)
        edge_beta = torch.pow(self.beta, torch.pow(1 - pred, self.gamma))
        bg_beta = torch.pow(self.beta, torch.pow(pred, self.gamma))
        loss = 0
        loss = loss - alpha * edge_beta * torch.log(pred) * gt
        loss = loss - (1 - alpha) * bg_beta * torch.log(1 - pred) * (1 - gt)
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
