import torch
import torch.nn as nn
import torch.utils.model_zoo


class penalty_bce_loss(nn.Module):

    def __init__(self):
        super(penalty_bce_loss, self).__init__()

    def forward(self, y_pred, y_true, pmap):
        B, C, W, H = y_pred.size()
        bce = -y_true * torch.log(y_pred + 1e-14) - (1 - y_true) * torch.log(
            1 - y_pred + 1e-14)
        bce = bce * pmap
        bce = torch.sum(bce) / (B * C * W * H)
        return bce


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
