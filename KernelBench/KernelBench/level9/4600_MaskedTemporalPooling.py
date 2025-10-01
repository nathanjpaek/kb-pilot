import torch
from typing import Optional
import torch.utils.data
import torch.nn


class MaskedTemporalPooling(torch.nn.Module):
    """
    Applies temporal pooling operations on masked inputs. For each pooling operation
    all masked values are ignored.
    """

    def __init__(self, method: 'str'):
        """
        method (str): the method of pooling to use. Options:
            'max': reduces temporal dimension to each valid max value.
            'avg': averages valid values in the temporal dimension.
            'sum': sums valid values in the temporal dimension.
            Note if all batch row elements are invalid, the temporal dimension is
            pooled to 0 values.
        """
        super().__init__()
        assert method in ('max', 'avg', 'sum')
        self._method = method

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None
        ) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        assert x.dim(
            ) == 3, 'Requires x shape (batch_size x seq_len x feature_dim)'
        b, t = x.shape[0], x.shape[1]
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool)
        if self._method == 'max':
            x[~mask, :] = float('-inf')
            invalid_first_dim = ~mask.view(b, -1).any(dim=-1)
            x[invalid_first_dim, :] = 0
            x = torch.max(x, dim=1)[0]
        elif self._method == 'avg':
            x = x * mask.unsqueeze(-1).float()
            mask = mask.view(b, t, -1).any(dim=-1)
            valid_lengths = mask.float().sum(dim=-1).int()
            x = x.sum(dim=1)
            x = x.div(valid_lengths.clamp(min=1).unsqueeze(-1).expand(x.
                size()).float())
        elif self._method == 'sum':
            x = x * mask.unsqueeze(-1).float()
            x = x.sum(dim=1)
        else:
            raise NotImplementedError(
                f"{self._method} not available options are: 'max', 'avg', 'sum'"
                )
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'method': 'max'}]
