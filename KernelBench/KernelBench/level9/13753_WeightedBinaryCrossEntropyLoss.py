import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.autograd


class WeightedBinaryCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor',
        weights: 'torch.Tensor'):
        """
        Args:
            input: (B, ...) float tensor.
                Predited logits for each class.
            target: (B, ...) float tensor.
                One-hot classification targets.
            weights: (B, ...) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        assert input.shape == target.shape
        assert input.shape == weights.shape
        loss = F.binary_cross_entropy_with_logits(input, target, reduction=
            'none') * weights
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
