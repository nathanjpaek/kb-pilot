import torch
import torch.nn as nn


class PerceptionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, results, targets):
        loss = 0.0
        for i, (ress, tars) in enumerate(zip(results, targets)):
            loss += self.l1loss(ress, tars)
        return loss / len(results)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
