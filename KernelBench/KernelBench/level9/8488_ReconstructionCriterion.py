import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionCriterion(nn.Module):
    """
    Here we calculate the criterion for -log p(x|z), we list two forms, the binary cross entropy form
    as well as the mse loss form
    """

    def __init__(self, x_sigma=1, bce_reconstruction=True):
        super(ReconstructionCriterion, self).__init__()
        self.x_sigma = x_sigma
        self.bce_reconstruction = bce_reconstruction

    def forward(self, x, x_reconstructed):
        batch_size = x.size(0)
        if self.bce_reconstruction:
            reconstruct_loss = F.binary_cross_entropy_with_logits(
                x_reconstructed, x, reduction='sum') / batch_size
        else:
            reconstruct_loss = F.mse_loss(torch.sigmoid(x_reconstructed), x,
                reduction='sum') / (2 * batch_size * self.x_sigma ** 2)
        return reconstruct_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
