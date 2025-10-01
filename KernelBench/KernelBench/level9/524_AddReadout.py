import torch
import torch.nn as nn


class AddReadout(nn.Module):
    """Handles readout operation when `readout` parameter is `add`. Removes `cls_token` or  `readout_token` from tensor and adds it to the rest of tensor"""

    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
