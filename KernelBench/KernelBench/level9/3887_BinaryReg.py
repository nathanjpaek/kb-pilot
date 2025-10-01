import torch
import torch.utils.data
import torch.nn as nn


class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred):
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=0.01)
        loss = (1.0 / diff).mean()
        return self.alpha * loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
