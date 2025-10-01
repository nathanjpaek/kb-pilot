import torch
import torch.utils.data
from torch import nn


class MySigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, confids, targets):
        bias = 1e-07
        num_classes = confids.shape[1]
        dtype = targets.dtype
        device = targets.device
        gamma = self.gamma
        alpha = self.alpha
        class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=
            device).unsqueeze(0)
        t = targets.unsqueeze(1)
        p = confids
        term1 = (1 - p) ** gamma * torch.log(p + bias)
        term2 = p ** gamma * torch.log(1 - p + bias)
        loss = -(t == class_range).float() * term1 * alpha - ((t !=
            class_range) * (t >= 0)).float() * term2 * (1 - alpha)
        return loss.sum() / confids.shape[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'gamma': 4, 'alpha': 4}]
