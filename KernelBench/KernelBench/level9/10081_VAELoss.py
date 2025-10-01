import torch
import torch.nn as nn


class VAELoss(nn.Module):

    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='sum')

    def forward(self, recon_x, x, mu, logvar):
        BCE = self.bce(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
