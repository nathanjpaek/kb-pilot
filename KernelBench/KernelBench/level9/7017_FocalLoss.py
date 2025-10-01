import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, reduce=True, gamma=1.5, alpha=0.7):
        super(FocalLoss, self).__init__()
        self.reduce = reduce
        self.gamma = gamma
        self.alpha = alpha

    def _get_weights(self, x, t):
        """
        Helper to get the weights for focal loss calculation
        """
        p = nn.functional.sigmoid(x)
        p_t = p * t + (1 - p) * (1 - t)
        alpha_t = self.alpha * t + (1 - self.alpha) * (1 - t)
        w = alpha_t * (1 - p_t).pow(self.gamma)
        return w

    def focal_loss(self, x, t):
        """
        Focal Loss cf. arXiv:1708.02002
        
        Args:
          x: (tensor) output from last layer of network
          t: (tensor) targets in [0,1]
          alpha: (float) class imbalance correction weight \\in (0,1)
          gamma: (float) amplification factor for uncertain classification
          
        Return:
          (tensor) focal loss.
        """
        weights = self._get_weights(x, t)
        return nn.functional.binary_cross_entropy_with_logits(x, t, weights,
            size_average=False, reduce=self.reduce)

    def forward(self, input, target):
        return self.focal_loss(input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
