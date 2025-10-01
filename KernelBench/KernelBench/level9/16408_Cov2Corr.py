import torch
import torch.nn as nn


class Cov2Corr(nn.Module):
    """Conversion from covariance matrix to correlation matrix."""

    def forward(self, covmat):
        """Convert.

        Parameters
        ----------
        covmat : torch.Tensor
            Covariance matrix of shape (n_samples, n_assets, n_assets).

        Returns
        -------
        corrmat : torch.Tensor
            Correlation matrix of shape (n_samples, n_assets, n_assets).

        """
        n_samples, n_assets, _ = covmat.shape
        stds = torch.sqrt(torch.diagonal(covmat, dim1=1, dim2=2))
        stds_ = stds.view(n_samples, n_assets, 1)
        corr = covmat / torch.matmul(stds_, stds_.permute(0, 2, 1))
        return corr


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
