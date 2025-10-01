import torch
from torch import nn
from torch.nn.modules.loss import *
from torch.nn.modules import *
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.backends


class PairwiseLoss(nn.Module):
    """Base class for pairwise loss functions.

    Pairwise approached looks at a pair of documents at a time in the loss function.
    Given a pair of documents the algorithm try to come up with the optimal ordering
    For that pair and compare it with the ground truth. The goal for the ranker is to
    minimize the number of inversions in ranker.

    Input space: pairs of documents (d1, d2)
    Output space: preferences (yes/no) for a given doc. pair
    """

    @staticmethod
    def _assert_equal_size(positive_score: 'torch.Tensor', negative_score:
        'torch.Tensor') ->None:
        if positive_score.size() != negative_score.size():
            raise ValueError(
                f'Shape mismatch: {positive_score.size()}, {negative_score.size()}'
                )

    def forward(self, positive_score: 'torch.Tensor', negative_score:
        'torch.Tensor'):
        raise NotImplementedError()


class BPRLoss(PairwiseLoss):
    """Bayesian Personalised Ranking loss function.

    It has been proposed in `BPRLoss\\: Bayesian Personalized Ranking from Implicit Feedback`_.

    .. _BPRLoss\\: Bayesian Personalized Ranking from Implicit Feedback:
        https://arxiv.org/pdf/1205.2618.pdf

    Args:
        gamma (float): Small value to avoid division by zero. Default: ``1e-10``.

    Example:

    .. code-block:: python

        import torch
        from catalyst.contrib.nn.criterion import recsys

        pos_score = torch.randn(3, requires_grad=True)
        neg_score = torch.randn(3, requires_grad=True)

        output = recsys.BPRLoss()(pos_score, neg_score)
        output.backward()
    """

    def __init__(self, gamma=1e-10) ->None:
        super().__init__()
        self.gamma = gamma

    def forward(self, positive_score: 'torch.Tensor', negative_score:
        'torch.Tensor') ->torch.Tensor:
        """Forward propagation method for the BPR loss.

        Args:
            positive_score: Tensor containing predictions for known positive items.
            negative_score: Tensor containing predictions for sampled negative items.

        Returns:
            computed loss
        """
        self._assert_equal_size(positive_score, negative_score)
        loss = -torch.log(self.gamma + torch.sigmoid(positive_score -
            negative_score))
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
