import torch
from torch import nn


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 0.0005
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        left = (self.upper - logits).relu() ** 2
        right = (logits - self.lower).relu() ** 2
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 -
            labels) * right)
        reconstruction_loss = self.mse(reconstructions.contiguous().view(
            images.shape), images)
        return (margin_loss + self.reconstruction_loss_scalar *
            reconstruction_loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
