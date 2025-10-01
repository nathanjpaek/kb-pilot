import torch
import torch.nn as nn


class InstancesAccuracy(nn.Module):
    """
    This class implements the accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: 'float'=0.5) ->None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        super(InstancesAccuracy, self).__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor', **
        kwargs) ->torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Accuracy
        """
        prediction = (prediction > self.threshold).float()
        correct_classified_elements = (prediction == label).float().sum()
        return correct_classified_elements / prediction.numel()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
