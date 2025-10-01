import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    """
    This class implements the lovasz hinge loss which is the continuous of the IoU for binary segmentation.
    Source: https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self) ->None:
        """
        Constructor method
        """
        super(LovaszHingeLoss, self).__init__()

    def _calc_grad(self, label_sorted: 'torch.Tensor') ->torch.Tensor:
        """
        Method computes the gradients of the sorted and flattened label
        :param label_sorted: (torch.Tensor) Sorted and flattened label of shape [n]
        :return: (torch.Tensor) Gradient tensor
        """
        label_sum = label_sorted.sum()
        intersection = label_sum - label_sorted.cumsum(dim=0)
        union = label_sum + (1 - label_sorted).cumsum(dim=0)
        iou = 1.0 - intersection / union
        iou[1:] = iou[1:] - iou[0:-1]
        return iou

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward pass computes the dice loss
        :param prediction: (torch.Tensor) Prediction
        :param label: (torch.Tensor) Label
        :return: (torch.Tensor) Dice loss value
        """
        prediction = prediction.flatten(start_dim=0)
        label = label.flatten(start_dim=0)
        signs = 2.0 * label - 1.0
        error = 1.0 - prediction * signs
        errors_sorted, permutation = torch.sort(error, dim=0, descending=True)
        label_sorted = label[permutation]
        grad = self._calc_grad(label_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
