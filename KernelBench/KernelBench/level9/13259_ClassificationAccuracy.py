import torch
import torch.nn as nn


class ClassificationAccuracy(nn.Module):
    """
    This class implements the classification accuracy computation. No gradients supported.
    """

    def __init__(self, threshold: 'float'=0.5) ->None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        super(ClassificationAccuracy, self).__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the accuracy score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Accuracy
        """
        correct_classified_elements = (prediction == label).float().sum()
        return correct_classified_elements / prediction.numel()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
