import torch
import torch.nn.functional as F


class focal_loss(torch.nn.Module):
    """
    Loss function for classification tasks  with
    large data imbalance. Focal loss (FL) is define as:
    FL(p_t) = -alpha*((1-p_t)^gamma))*log(p_t),
    where p_t is a cross-entropy loss for binary classification.
    For more details, see https://arxiv.org/abs/1708.02002.

    Args:
        alpha (float):
            "balance" coefficient,
        gamma (float):
            "focusing" parameter (>=0),
        with_logits (bool):
            indicates if the sigmoid operation was applied
            at the end of a neural network's forward path.
    """

    def __init__(self, alpha: 'int'=0.5, gamma: 'int'=2, with_logits:
        'bool'=True) ->None:
        """
        Parameter initialization
        """
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = with_logits

    def forward(self, prediction: 'torch.Tensor', labels: 'torch.Tensor'):
        """
        Calculates loss
        """
        if self.logits:
            CE_loss = F.binary_cross_entropy_with_logits(prediction, labels)
        else:
            CE_loss = F.binary_cross_entropy(prediction, labels)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return F_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
