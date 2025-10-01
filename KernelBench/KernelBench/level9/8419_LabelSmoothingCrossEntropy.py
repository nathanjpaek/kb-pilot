import torch
import torch._C
import torch.serialization
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1, loss_weight=1.0, loss_name='loss_ce'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, x: 'torch.Tensor', target: 'torch.Tensor', weight=
        None, avg_factor=None, reduction_override=None, **kwargs
        ) ->torch.Tensor:
        x = x.permute(1, 0, 2, 3).flatten(1).transpose(0, 1)
        target = target.flatten(0)
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (self.confidence * nll_loss + self.smoothing * smooth_loss
            ) * self.loss_weight
        return loss.mean()

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


def get_inputs():
    return [torch.rand([64, 4, 4, 4]), torch.ones([1024], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
