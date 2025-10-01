from torch.nn import Module
import torch
from typing import cast
from torch.nn.modules import Module


class QRLoss(Module):
    """The QR (forward) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

    def forward(self, probs: 'torch.Tensor', target: 'torch.Tensor'
        ) ->torch.Tensor:
        """Computes the QR (forwards) loss on prior.

        Args:
            probs: probabilities of predictions, expected shape B x C x H x W.
            target: prior probabilities, expected shape B x C x H x W.

        Returns:
            qr loss
        """
        q = probs
        q_bar = q.mean(dim=(0, 2, 3))
        qbar_log_S = (q_bar * torch.log(q_bar)).sum()
        q_log_p = torch.einsum('bcxy,bcxy->bxy', q, torch.log(target)).mean()
        loss = qbar_log_S - q_log_p
        return cast(torch.Tensor, loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
