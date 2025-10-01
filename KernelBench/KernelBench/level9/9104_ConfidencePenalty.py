import torch
import torch.utils.data
from torch import nn


class ConfidencePenalty(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, epsilon=1.0, device='cpu'):
        super(ConfidencePenalty, self).__init__()
        self.epsilon = epsilon
        self.device = device
        self.logsoftmax = nn.LogSoftmax()
        self.baseloss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        entropy = -torch.sum(inputs * torch.log(inputs.clamp(min=1e-08, max
            =1.0)), dim=1)
        confidence_penalty = -self.epsilon * entropy.mean(0)
        loss_xent = self.baseloss(inputs, targets)
        loss = loss_xent + confidence_penalty
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
