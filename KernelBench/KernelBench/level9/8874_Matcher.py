import math
import torch
import torch.nn as nn


class Matcher(nn.Module):
    """
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    """

    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.left_linear = nn.Linear(n_hid, n_hid)
        self.right_linear = nn.Linear(n_hid, n_hid)
        self.sqrt_hd = math.sqrt(n_hid)
        self.cache = None

    def forward(self, x, y, infer=False, pair=False):
        ty = self.right_linear(y)
        if infer:
            """
                During testing, we will consider millions or even billions of nodes as candidates (x).
                It's not possible to calculate them again for different query (y)
                Since the model is fixed, we propose to cache them, and dirrectly use the results.
            """
            if self.cache is not None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0, 1))
        return res / self.sqrt_hd

    def __repr__(self):
        return '{}(n_hid={})'.format(self.__class__.__name__, self.n_hid)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_hid': 4}]
