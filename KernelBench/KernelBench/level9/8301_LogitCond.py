import torch
import torch.nn as nn


class LogitCond(nn.Module):
    """
    from the softmax outputs, decides whether the samples are above or below threshold.
    """

    def __init__(self, thres=1.0):
        super(LogitCond, self).__init__()
        self.thres = thres
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        logits = self.softmax(outputs)
        max_logits, _ = torch.max(logits, dim=1)
        cond_up = max_logits > self.thres
        cond_down = max_logits <= self.thres
        return cond_up, cond_down


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
