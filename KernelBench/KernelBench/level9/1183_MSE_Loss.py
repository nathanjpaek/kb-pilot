import torch
import torch.nn as nn


class MSE_Loss(nn.Module):

    def __init__(self, sum_dim=None, sqrt=False, dimension_warn=0):
        super().__init__()
        self.sum_dim = sum_dim
        self.sqrt = sqrt
        self.dimension_warn = dimension_warn

    def forward(self, x, y):
        assert x.shape == y.shape
        if self.sum_dim:
            mse_loss = torch.sum((x - y) ** 2, dim=self.sum_dim)
        else:
            mse_loss = torch.sum((x - y) ** 2)
        if self.sqrt:
            mse_loss = torch.sqrt(mse_loss)
        mse_loss = torch.sum(mse_loss) / mse_loss.nelement()
        if len(mse_loss.shape) > self.dimension_warn:
            raise ValueError(
                'The shape of mse loss should be a scalar, but you can skip thiserror by change the dimension_warn explicitly.'
                )
        return mse_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
