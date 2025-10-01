import torch
import torch.nn as nn
import torch.utils.data


class UnbalancedLoss(nn.Module):
    NUM_LABELS = 2

    def __init__(self):
        super().__init__()
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, logits, label):
        return self.crit(logits, label)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
