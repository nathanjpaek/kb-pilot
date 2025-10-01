import torch
import torch.nn as nn


class IoU(nn.Module):
    """
    This class implements the IoU for validation. Not gradients supported.
    """

    def __init__(self, threshold: 'float'=0.5) ->None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        super(IoU, self).__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor', **
        kwargs) ->torch.Tensor:
        """
        Forward pass computes the IoU score
        :param prediction: (torch.Tensor) Prediction of all shapes
        :param label: (torch.Tensor) Label of all shapes
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) IoU score
        """
        prediction = (prediction > self.threshold).float()
        intersection = (prediction + label == 2.0).sum()
        union = (prediction + label >= 1.0).sum()
        return intersection / (union + 1e-10)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
