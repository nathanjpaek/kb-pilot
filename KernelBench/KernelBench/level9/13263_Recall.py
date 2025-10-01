import torch
import torch.nn as nn


class Recall(nn.Module):
    """
    This class implements the recall score. No gradients supported.
    """

    def __init__(self, threshold: 'float'=0.5) ->None:
        """
        Constructor method
        :param threshold: (float) Threshold to be applied
        """
        super(Recall, self).__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor', **
        kwargs) ->torch.Tensor:
        """
        Forward pass computes the recall score
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :param kwargs: Key word arguments (not used)
        :return: (torch.Tensor) Recall score
        """
        prediction = (prediction > self.threshold).float()
        true_positive_elements = ((prediction == 1.0).float() + (label == 
            1.0) == 2.0).float()
        false_negative_elements = ((prediction == 0.0).float() + (label == 
            1.0) == 2.0).float()
        return true_positive_elements.sum() / ((true_positive_elements +
            false_negative_elements).sum() + 1e-10)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
