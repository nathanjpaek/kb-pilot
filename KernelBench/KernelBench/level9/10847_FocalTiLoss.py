import torch
import torch.nn as nn


class FocalTiLoss(nn.Module):

    def __init__(self, alpha=0.7, beta=0.4, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = 1e-06

    def forward(self, output, target):
        output = output.float()
        target = target.float()
        pi = output.contiguous().view(-1)
        gi = target.contiguous().view(-1)
        p_ = 1 - pi
        g_ = 1 - gi
        intersection = torch.dot(pi, gi)
        inter_alpha = torch.dot(p_, gi)
        inter_beta = torch.dot(g_, pi)
        ti = (intersection + self.eps) / (intersection + self.alpha *
            inter_alpha + self.beta * inter_beta + self.eps)
        loss = torch.mean(torch.pow(1 - ti, self.gamma))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
