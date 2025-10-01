import torch
import torch.nn as nn


def log_minus_sigmoid(x):
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))
        ) + 0.5 * torch.clamp(x, min=0, max=0)


def log_sigmoid(x):
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))
        ) + 0.5 * torch.clamp(x, min=0, max=0)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        pos_log_sig = log_sigmoid(input)
        neg_log_sig = log_minus_sigmoid(input)
        prob = torch.sigmoid(input)
        pos_weight = torch.pow(1 - prob, self.gamma)
        neg_weight = torch.pow(prob, self.gamma)
        loss = -(target * pos_weight * pos_log_sig + (1 - target) *
            neg_weight * neg_log_sig)
        avg_weight = target * pos_weight + (1 - target) * neg_weight
        loss /= avg_weight.mean()
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
