import torch
import torch.nn as nn


class TreeLSTM(nn.Module):
    """
    Implementation of the Tree-LSTM model:
    https://arxiv.org/pdf/1503.00075.pdf
    """

    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.left = nn.Linear(num_units, 5 * num_units)
        self.right = nn.Linear(num_units, 5 * num_units)

    def forward(self, left_in, right_in):
        lstm_in = self.left(left_in[0])
        lstm_in += self.right(right_in[0])
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] + f2.sigmoid(
            ) * right_in[1]
        h = o.sigmoid() * c.tanh()
        return h, c


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_units': 4}]
