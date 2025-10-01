import torch
from torch import nn


class ELBOLoss(nn.Module):

    def __init__(self):
        super(ELBOLoss, self).__init__()
        self.recons_loss = nn.BCELoss(reduction='sum')

    def forward(self, reconstruction, x, mu, log_var):
        loss = -self.recons_loss(reconstruction, x)
        KL_loss = 0.5 * torch.sum(-1 - log_var + mu ** 2 + log_var.exp())
        return -(loss - KL_loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
