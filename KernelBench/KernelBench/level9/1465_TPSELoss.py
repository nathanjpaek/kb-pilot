import torch
import torch.utils.data
from torch import nn


class TPSELoss(nn.Module):

    def __init__(self):
        super(TPSELoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, model_output):
        gst_embed, tpse_embed = model_output[4], model_output[5]
        gst_embed = gst_embed.detach()
        loss = self.loss(tpse_embed, gst_embed)
        return loss


def get_inputs():
    return [torch.rand([6, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
