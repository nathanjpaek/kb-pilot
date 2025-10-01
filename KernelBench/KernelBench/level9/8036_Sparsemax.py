import torch
import torch as th
import torch.nn as nn


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=-1):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, x):
        """Forward function.
        Args:
            x (th.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            th.Tensor: [batch_size x number_of_logits] Output tensor
        """
        x = x.transpose(0, self.dim)
        original_size = x.size()
        x = x.reshape(x.size(0), -1)
        x = x.transpose(0, 1)
        dim = 1
        number_of_logits = x.size(dim)
        x = x - th.max(x, dim=dim, keepdim=True)[0].expand_as(x)
        zs = th.sort(x, dim=dim, descending=True)[0]
        rg = th.arange(start=1, end=number_of_logits + 1, step=1, device=x.
            device, dtype=x.dtype).view(1, -1)
        rg = rg.expand_as(zs)
        bound = 1 + rg * zs
        cumulative_sum_zs = th.cumsum(zs, dim)
        is_gt = th.gt(bound, cumulative_sum_zs).type(x.type())
        k = th.max(is_gt * rg, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (th.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(x)
        self.output = th.max(th.zeros_like(x), x - taus)
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output

    def backward(self, x):
        """Backward function."""
        dim = 1
        nonzeros = th.ne(self.output, 0)
        x_sum = th.sum(x * nonzeros, dim=dim) / th.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (x - x_sum.expand_as(x))
        return self.grad_input

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
