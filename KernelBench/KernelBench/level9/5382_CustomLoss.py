import torch
import torch.nn as nn


class CustomLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        gamma = 0.5
        C4 = 10
        gb_hat = outputs[:, :, :34]
        rb_hat = outputs[:, :, 34:68]
        gb = targets[:, :, :34]
        rb = targets[:, :, 34:68]
        """
        total_loss=0
        for i in range(500):
            total_loss += (torch.sum(torch.pow((torch.pow(gb[:,i,:],gamma) - torch.pow(gb_hat[:,i,:],gamma)),2)))              + C4*torch.sum(torch.pow(torch.pow(gb[:,i,:],gamma) - torch.pow(gb_hat[:,i,:],gamma),4))              + torch.sum(torch.pow(torch.pow((1-rb[:,i,:]),gamma)-torch.pow((1-rb_hat[:,i,:]),gamma),2))
        return total_loss
        """
        return torch.mean(torch.pow(torch.pow(gb, gamma) - torch.pow(gb_hat,
            gamma), 2)) + C4 * torch.mean(torch.pow(torch.pow(gb, gamma) -
            torch.pow(gb_hat, gamma), 4)) + torch.mean(torch.pow(torch.pow(
            1 - rb, gamma) - torch.pow(1 - rb_hat, gamma), 2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
