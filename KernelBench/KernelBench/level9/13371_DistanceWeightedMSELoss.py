import torch
from torch import nn
import torch.utils.data


class DistanceWeightedMSELoss(nn.Module):
    """Weighted MSE loss for signed euclidean distance transform targets.

    By setting ``fg_weight`` to a high value, the errors in foreground
    regions are more strongly penalized.
    If ``fg_weight=1``, this loss is equivalent to ``torch.nn.MSELoss``.

    Requires that targets are transformed with
    :py:class:`elektronn3.data.transforms.DistanceTransformTarget`

    Per-pixel weights are assigned on the targets as follows:
    - each foreground pixel is weighted by ``fg_weight``
    - each background pixel is weighted by 1.
    """

    def __init__(self, fg_weight=100.0, mask_borders=40):
        super().__init__()
        self.fg_weight = fg_weight
        self.mask_borders = mask_borders

    def forward(self, output, target):
        mse = nn.functional.mse_loss(output, target, reduction='none')
        with torch.no_grad():
            weight = torch.ones_like(target)
            weight[target <= 0] = self.fg_weight
            if self.mask_borders is not None:
                o = self.mask_borders
                weight[:, :, :o, :o] = 0.0
                weight[:, :, target.shape[-2] - o:, target.shape[-1] - o:
                    ] = 0.0
        return torch.mean(weight * mse)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
