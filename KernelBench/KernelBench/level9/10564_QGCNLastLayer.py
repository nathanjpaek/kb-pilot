from torch.nn import Module
import torch
from torch.nn import Linear


class QGCNLastLayer(Module):

    def __init__(self, left_in_dim, right_in_dim, out_dim):
        super(QGCNLastLayer, self).__init__()
        self._left_linear = Linear(left_in_dim, 1)
        self._right_linear = Linear(right_in_dim, out_dim)
        self._gpu = False

    def forward(self, A, x0, x1):
        x1_A = torch.matmul(x1.permute(0, 2, 1), A)
        W2_x1_A = self._left_linear(x1_A.permute(0, 2, 1))
        W2_x1_A_x0 = torch.matmul(W2_x1_A.permute(0, 2, 1), x0)
        W2_x1_A_x0_W3 = self._right_linear(W2_x1_A_x0)
        return W2_x1_A_x0_W3.squeeze(dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'left_in_dim': 4, 'right_in_dim': 4, 'out_dim': 4}]
