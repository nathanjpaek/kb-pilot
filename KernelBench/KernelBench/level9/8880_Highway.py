import torch
import torch.nn as nn
import torch.utils.data


class Highway(nn.Linear):
    """
    :param input_dim: Scalar.
    :param drop_rate: Scalar. dropout rate
    
    """

    def __init__(self, input_dim, drop_rate=0.0):
        self.drop_rate = drop_rate
        super(Highway, self).__init__(input_dim, input_dim * 2)
        self.drop_out = nn.Dropout(self.drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        """
        :param x: (N, T, input_dim) Tensor.

        Returns:
            y: (N, T, input_dim) Tensor.

        """
        y = super(Highway, self).forward(x)
        h, y_ = y.chunk(2, dim=-1)
        h = torch.sigmoid(h)
        y_ = torch.relu(y_)
        y_ = h * y_ + (1 - h) * x
        y_ = self.drop_out(y_) if self.drop_out is not None else y_
        return y_


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
