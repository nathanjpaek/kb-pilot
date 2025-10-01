import torch
import torch.nn as nn


class SAM_Loss(nn.Module):

    def __init__(self):
        super(SAM_Loss, self).__init__()

    def forward(self, output, label):
        ratio = torch.sum((output + 1e-08).mul(label + 1e-08), dim=1
            ) / torch.sqrt(torch.sum((output + 1e-08).mul(output + 1e-08),
            dim=1) * torch.sum((label + 1e-08).mul(label + 1e-08), dim=1))
        angle = torch.acos(ratio)
        return torch.mean(angle)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
