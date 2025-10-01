import torch
import torch.nn.functional as F
from torch import nn


class TorchFocalLoss(nn.Module):
    """Implementation of Focal Loss[1]_ modified from Catalyst [2]_ .

    Arguments
    ---------
    gamma : :class:`int` or :class:`float`
        Focusing parameter. See [1]_ .
    alpha : :class:`int` or :class:`float`
        Normalization factor. See [1]_ .

    References
    ----------
    .. [1] https://arxiv.org/pdf/1708.02002.pdf
    .. [2] https://catalyst-team.github.io/catalyst/
    """

    def __init__(self, gamma=2, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, targets):
        """Calculate the loss function between `outputs` and `targets`.

        Arguments
        ---------
        outputs : :class:`torch.Tensor`
            The output tensor from a model.
        targets : :class:`torch.Tensor`
            The training target.

        Returns
        -------
        loss : :class:`torch.Variable`
            The loss value.
        """
        if targets.size() != outputs.size():
            raise ValueError(
                f'Targets and inputs must be same size. Got ({targets.size()}) and ({outputs.size()})'
                )
        max_val = (-outputs).clamp(min=0)
        log_ = ((-max_val).exp() + (-outputs - max_val).exp()).log()
        loss = outputs - outputs * targets + max_val + log_
        invprobs = F.logsigmoid(-outputs * (targets * 2.0 - 1.0))
        loss = self.alpha * (invprobs * self.gamma).exp() * loss
        return loss.sum(dim=-1).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
