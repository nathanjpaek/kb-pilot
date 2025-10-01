import math
import torch
from torch import nn
import torch.nn.functional as F


class IcosahedronUnpool(nn.Module):
    """Isocahedron Unpooling, consists in adding 1 values to match the desired un pooling size
    """

    def forward(self, x):
        """Forward calculates the subset of pixels that will result from the unpooling kernel_size and then adds 1 valued pixels to match this size

        Args:
            x (:obj:`torch.tensor`) : [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`]: [batch x pixels unpooled x features]
        """
        M = x.size(1)
        order = int(math.log((M - 2) / 10) / math.log(4))
        unpool_order = order + 1
        additional_pixels = int(10 * math.pow(4, unpool_order) + 2)
        subset_pixels_add = additional_pixels - M
        return F.pad(x, (0, 0, 0, subset_pixels_add, 0, 0), 'constant', value=1
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
