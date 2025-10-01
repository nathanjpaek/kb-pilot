import torch
import torch.nn as nn
import torch.nn.functional as F


class CocoLinear(nn.Module):
    """Congenerous Cosine linear module (for CoCo loss)

        Parameters
        ----------
        nfeat : int
            Embedding dimension
        nclass : int
            Number of classes
        alpha : float
            Scaling factor used in embedding L2-normalization
        """

    def __init__(self, nfeat, nclass, alpha):
        super(CocoLinear, self).__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(nclass, nfeat))

    def forward(self, x):
        """Apply the angular margin transformation

        Parameters
        ----------
        x : `torch.Tensor`
            an embedding batch
        Returns
        -------
        fX : `torch.Tensor`
            logits after the congenerous cosine transformation
        """
        cnorm = F.normalize(self.centers)
        xnorm = self.alpha * F.normalize(x)
        logits = torch.matmul(xnorm, torch.transpose(cnorm, 0, 1))
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nclass': 4, 'alpha': 4}]
