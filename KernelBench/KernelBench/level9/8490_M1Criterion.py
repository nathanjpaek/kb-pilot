import torch
import torch.nn as nn
import torch.nn.functional as F


class M1Criterion(nn.Module):

    def __init__(self, x_sigma=1, bce_reconstruction=True):
        super(M1Criterion, self).__init__()
        self.x_sigma = x_sigma
        self.bce_reconstruction = bce_reconstruction

    def forward(self, x, x_reconstructed, M1_mean, M1_log_sigma):
        batch_size = x.size(0)
        if self.bce_reconstruction:
            reconstruct_loss = F.binary_cross_entropy_with_logits(
                x_reconstructed, x, reduction='sum') / batch_size
        else:
            reconstruct_loss = F.mse_loss(torch.sigmoid(x_reconstructed), x,
                reduction='sum') / (2 * batch_size * self.x_sigma ** 2)
        M1_mean_sq = M1_mean * M1_mean
        M1_log_sigma_sq = 2 * M1_log_sigma
        M1_sigma_sq = torch.exp(M1_log_sigma_sq)
        M1_continuous_kl_loss = 0.5 * torch.sum(M1_mean_sq + M1_sigma_sq -
            M1_log_sigma_sq - 1) / batch_size
        return reconstruct_loss, M1_continuous_kl_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
