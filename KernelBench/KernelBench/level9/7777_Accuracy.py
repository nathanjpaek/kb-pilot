import torch
import torch.nn as nn


class Accuracy(nn.Module):
    """
    This class implements the accuracy metric.
    """

    def __init__(self) ->None:
        """
        Constructor method
        """
        super(Accuracy, self).__init__()

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the accuracy metric
        :param prediction: (torch.Tensor) Prediction of the shape [batch size, classes] (one-hot)
        :param label: (torch.Tensor) Classification label of the shape [batch size]
        :return: (torch.Tensor) Accuracy metric
        """
        prediction = prediction.argmax(dim=-1)
        accuracy = (prediction == label).sum() / float(prediction.shape[0])
        return accuracy


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
