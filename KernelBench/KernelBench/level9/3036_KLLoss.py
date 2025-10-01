import torch
import torch.nn as nn
import torch.cuda
from torch import linalg as linalg


class BaseMeasure(nn.Module):
    """
    """
    NAME: 'str' = NotImplemented
    REFERENCE: 'str' = None
    BIGGER_IS_BETTER = False
    OPT_VALUE = 0.0

    def __init__(self, device):
        """
        Args:
            device ():
        """
        super(BaseMeasure, self).__init__()
        self.device = device
        self

    def forward(self, pred: 'torch.Tensor', target: 'torch.Tensor'):
        """
        Args:
            pred ():
            target ():
        Returns:
        """
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError(f'{self.NAME} expects 5-D inputs!')
        value = self.criterion(pred, target)
        return value.sum(dim=(4, 3, 2)).mean(dim=1).mean(dim=0)

    def reshape_clamp(self, pred: 'torch.Tensor', target: 'torch.Tensor'):
        """
        Args:
            pred ():
            target ():
        Returns:
        """
        if pred.ndim != 5 or target.ndim != 5:
            raise ValueError(f'{self.NAME} expects 5-D inputs!')
        pred = pred.reshape(-1, *pred.shape[2:])
        pred = ((pred + 1) / 2).clamp_(min=0.0, max=1.0)
        target = target.reshape(-1, *target.shape[2:])
        target = ((target + 1) / 2).clamp_(min=0.0, max=1.0)
        return pred, target

    @classmethod
    def to_display(cls, x):
        """
        Args:
            x ():
        Returns:
        """
        return x


class KLLoss(BaseMeasure):
    """
    KL-Divergence loss function
    """
    NAME = 'KL-Divergence (KL)'

    def __init__(self, device):
        super(KLLoss, self).__init__(device)

    def criterion(self, mu1, logvar1, mu2, logvar2):
        """ Computing the KL-Divergence between two Gaussian distributions """
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2
            ) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld

    def forward(self, mu1, logvar1, mu2, logvar2):
        """ Computing the KL-Divergence between two Gaussian distributions """
        value = self.criterion(mu1, logvar1, mu2, logvar2)
        return value.sum(dim=-1).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0}]
