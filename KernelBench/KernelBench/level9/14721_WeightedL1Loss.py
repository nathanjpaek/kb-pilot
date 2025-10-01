import torch
import numpy as np
import torch.nn as nn


class WeightedL1Loss(nn.Module):

    def __init__(self, code_weights: 'list'=None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        self.code_weights = code_weights
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor',
        weights: 'torch.Tensor'=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)
        loss = torch.abs(diff)
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1
                ] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
