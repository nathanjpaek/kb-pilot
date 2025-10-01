import torch
import torch.nn as nn
import torch.nn.functional as F


class ConLoss(nn.Module):

    def __init__(self, device, margin=2):
        super(ConLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, output1, output2, label):
        diff = F.pairwise_distance(output1, output2)
        loss = label * torch.square(diff) + (1 - label) * torch.square(torch
            .clamp(self.margin - diff, min=0))
        return torch.mean(loss)

    def distance(self, output1, output2):
        diff = F.pairwise_distance(output1, output2)
        return diff


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0}]
