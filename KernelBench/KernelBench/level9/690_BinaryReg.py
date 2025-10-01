import torch
import torch.nn as nn
import torch.utils.data


class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred):
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=0.01)
        loss = 1.0 / diff.sum()
        return self.alpha * loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
