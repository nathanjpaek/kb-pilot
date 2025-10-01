import torch
from torch import nn


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(1, n_class + 1, dtype=target.dtype, device
            =target.device).unsqueeze(0)
        t = target.unsqueeze(1)
        p = torch.sigmoid(out)
        gamma = self.gamma
        alpha = self.alpha
        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)
        loss = -(t == class_ids).float() * alpha * term1 - ((t != class_ids
            ) * (t >= 0)).float() * (1 - alpha) * term2
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'gamma': 4, 'alpha': 4}]
