import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data
import torch.optim
import torch.utils.data.distributed


class VAE_Kl_Loss(nn.Module):

    def __init__(self, if_print=False):
        super(VAE_Kl_Loss, self).__init__()
        self.if_print = if_print

    def forward(self, means, variances):
        loss = self.standard_KL_loss(means, variances)
        if self.if_print:
            None
        return loss

    def standard_KL_loss(self, means, variances):
        loss_KL = torch.mean(torch.sum(0.5 * (means ** 2 + torch.exp(
            variances) - variances - 1), dim=1))
        return loss_KL


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
