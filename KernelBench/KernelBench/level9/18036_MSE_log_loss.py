import torch
import torch.nn as nn
import torch.utils.data
import torch.optim


class MSE_log_loss(nn.Module):

    def __init__(self):
        super(MSE_log_loss, self).__init__()

    def forward(self, prediction, gt):
        prediction = torch.clamp(prediction, min=0)
        err = torch.log(prediction + 1e-06) - torch.log(gt + 1e-06)
        mask = (gt > 0).detach()
        mae_log_loss = torch.mean(err[mask] ** 2)
        return mae_log_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
