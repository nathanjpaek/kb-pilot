import torch
import torch.nn as nn


class AttentionCollapse(nn.Module):
    """Collapsing over the channels with attention.

    Parameters
    ----------
    n_channels : int
        Number of input channels.

    Attributes
    ----------
    affine : nn.Module
        Fully connected layer performing linear mapping.

    context_vector : nn.Module
        Fully connected layer encoding direction importance.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.affine = nn.Linear(n_channels, n_channels)
        self.context_vector = nn.Linear(n_channels, 1, bias=False)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_channels, n_assets)`.

        """
        n_samples, _n_channels, _lookback, _n_assets = x.shape
        res_list = []
        for i in range(n_samples):
            inp_single = x[i].permute(2, 1, 0)
            tformed = self.affine(inp_single)
            w = self.context_vector(tformed)
            scaled_w = torch.nn.functional.softmax(w, dim=1)
            weighted_sum = (inp_single * scaled_w).mean(dim=1)
            res_list.append(weighted_sum.permute(1, 0))
        return torch.stack(res_list, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
