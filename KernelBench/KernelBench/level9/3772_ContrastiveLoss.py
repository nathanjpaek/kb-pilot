import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, weight):
        pairdist = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean(weight * torch.pow(pairdist, 2) + (1 -
            weight) * torch.pow(torch.clamp(self.margin - pairdist, min=0.0
            ), 2))
        return contrastive_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
