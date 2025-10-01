import torch
import torch.nn as nn
import torch.optim


class FocalLoss(torch.nn.Module):
    """Sigmoid focal cross entropy loss.
  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
    Args:
      gamma: exponent of the modulating factor (1 - p_t)^gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives,
           with alpha in [0, 1] for class 1 and 1-alpha for class 0. 
           In practice alpha may be set by inverse class frequency,
           so that for a low number of positives, its weight is high.
    """
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self.BCEWithLogits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, prediction_tensor, target_tensor):
        """Compute loss function.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets.
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
        per_entry_cross_ent = self.BCEWithLogits(prediction_tensor,
            target_tensor)
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = target_tensor * prediction_probabilities + (1 - target_tensor
            ) * (1 - prediction_probabilities)
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 -
                target_tensor) * (1 - self._alpha)
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
            per_entry_cross_ent)
        return torch.mean(focal_cross_entropy_loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
