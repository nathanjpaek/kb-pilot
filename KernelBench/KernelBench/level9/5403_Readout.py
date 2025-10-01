from torch.nn import Module
import torch
import torch.utils.data


def aggregate(x, dim, aggr='add', mask=None, keepdim=False):
    """
    Args:
        x:    (..., A, ..., F), Features to be aggregated.
        mask: (..., A, ...)
    Returns:
        (...,  , ..., F), if keepdim == False
        (..., 1, ..., F), if keepdim == True
    """
    assert aggr in ('add', 'mean')
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    y = torch.sum(x, dim=dim, keepdim=keepdim)
    if aggr == 'mean':
        if mask is not None:
            n = torch.sum(mask, dim=dim, keepdim=keepdim)
            n = torch.max(n, other=torch.ones_like(n))
        else:
            n = x.size(dim)
        y = y / n
    return y


def readout(x, mask, aggr='add'):
    """
    Args:
        x:    (B, N_max, F)
        mask: (B, N_max)
    Returns:
        (B, F)
    """
    return aggregate(x=x, dim=1, aggr=aggr, mask=mask, keepdim=False)


class Readout(Module):

    def __init__(self, aggr='add'):
        super().__init__()
        assert aggr in ('add', 'mean')
        self.aggr = aggr

    def forward(self, x, mask):
        return readout(x, mask=mask, aggr=self.aggr)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
