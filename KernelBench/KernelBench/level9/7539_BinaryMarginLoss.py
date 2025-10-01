import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class BinaryMarginLoss(nn.Module):

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, output):
        return torch.logaddexp(torch.tensor([1.0], device=output.device), 
            self.margin - output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
