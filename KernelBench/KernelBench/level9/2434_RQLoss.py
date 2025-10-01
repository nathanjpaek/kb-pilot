from torch.nn import Module
import torch
from typing import cast
from torch.nn.modules import Module
import torch.nn.functional as F


class RQLoss(Module):
    """The RQ (backwards) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

    def forward(self, probs: 'torch.Tensor', target: 'torch.Tensor'
        ) ->torch.Tensor:
        """Computes the RQ (backwards) loss on prior.

        Args:
            probs: probabilities of predictions, expected shape B x C x H x W
            target: prior probabilities, expected shape B x C x H x W

        Returns:
            qr loss
        """
        q = probs
        z = q / q.norm(p=1, dim=(0, 2, 3), keepdim=True).clamp_min(1e-12
            ).expand_as(q)
        r = F.normalize(z * target, p=1, dim=1)
        loss = torch.einsum('bcxy,bcxy->bxy', r, torch.log(r) - torch.log(q)
            ).mean()
        return cast(torch.Tensor, loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
