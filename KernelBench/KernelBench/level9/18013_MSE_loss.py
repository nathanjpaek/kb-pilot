import torch
import torch.nn as nn
import torch.utils.data
import torch.optim


class MSE_loss(nn.Module):

    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, prediction, gt, epoch=0):
        err = prediction[:, 0:1] - gt
        mask = (gt > 0).detach()
        mse_loss = torch.mean(err[mask] ** 2)
        return mse_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
