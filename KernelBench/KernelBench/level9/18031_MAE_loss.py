import torch
import torch.nn as nn
import torch.utils.data
import torch.optim


class MAE_loss(nn.Module):

    def __init__(self):
        super(MAE_loss, self).__init__()

    def forward(self, prediction, gt, epoch=0):
        prediction = prediction[:, 0:1]
        abs_err = torch.abs(prediction - gt)
        mask = (gt > 0).detach()
        mae_loss = torch.mean(abs_err[mask])
        return mae_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
