import torch
from torch import nn
import torch.nn.functional as F


class HealpixAvgPool(nn.AvgPool1d):
    """Healpix Average pooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch

        Arguments:
            x (:obj:`torch.tensor`): [batch x pixels x features]

        Returns:
            [:obj:`torch.tensor`] : [batch x pooled pixels x features]
        """
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, self.kernel_size)
        x = x.permute(0, 2, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
