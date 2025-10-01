import torch
import torch.nn.functional
import torch.nn as nn


class AttMSEloss(nn.Module):

    def __init__(self):
        super(AttMSEloss, self).__init__()

    def forward(self, x_org, y_mask, att):
        loss_att = ((x_org * y_mask[:, 1, ...].unsqueeze(dim=1) - att) ** 2
            ).mean()
        loss_att = torch.clamp(loss_att, max=30)
        return 10 * loss_att


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
