import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Wang(nn.Module):
    """Neural network model for linear combination of EDU scores.

    """

    def __init__(self, nrels):
        """Class constructor.

        Args:
          nrels (int): total number of relations

        """
        super(Wang, self).__init__()
        d = np.ones((nrels, 1), dtype=np.float32)
        d[0] = 0
        self.d = nn.Parameter(torch.tensor(d))
        self.b = nn.Parameter(torch.tensor(0.5))

    def forward(self, rel_indices, x):
        rel_coeffs = self.d[rel_indices]
        ret = torch.sum(rel_coeffs * x, dim=1) + self.b
        return F.softmax(ret, dim=-1)


def get_inputs():
    return [torch.ones([4], dtype=torch.int64), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nrels': 4}]
