import torch
import torch.nn as nn


class focal_BCELoss(nn.Module):

    def __init__(self, alpha=10, gamma=2):
        super(focal_BCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, eps=1e-07):
        input = torch.clamp(input, eps, 1 - eps)
        loss = -(target * torch.log(input)) * torch.exp(self.alpha * (1 -
            input) ** self.gamma) - (1 - target) * torch.log(1 - input)
        final_loss = torch.mean(loss)
        return final_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
