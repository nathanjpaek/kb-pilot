import torch
import torch.nn as nn


class FocalLossMultiClass(nn.Module):
    """
    Implementation of the multi class focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: 'float'=0.25, gamma: 'float'=2.0) ->None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}, alpha={}, gamma={}'.format(self.__class__.__name__,
            self.alpha, self.gamma)

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        cross_entropy_loss = -(label * torch.log(prediction.clamp(min=1e-12))
            ).sum(dim=0)
        focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
        loss = ((1.0 - focal_factor) ** self.gamma * cross_entropy_loss *
            self.alpha).sum(dim=0).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
