import torch
import torch.nn as nn


class WasLoss(nn.Module):

    def __init__(self):
        super(WasLoss, self).__init__()
        self.MSEls = torch.nn.BCEWithLogitsLoss()

    def forward(self, true_data, fake_data):
        SLX, _ = torch.sort(true_data, 0)
        SLG, _ = torch.sort(fake_data, 0)
        return self.MSEls(SLG - SLX, torch.ones_like(SLX))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
