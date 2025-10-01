import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F


class StochasticGate(nn.Module):
    """Stochastically merges features from two levels 
    with varying size of the receptive field
    """

    def __init__(self):
        super(StochasticGate, self).__init__()
        self._mask_drop = None

    def forward(self, x1, x2, alpha_rate=0.3):
        """Stochastic Gate (SG)

        SG stochastically mixes deep and shallow features
        at training time and deterministically combines 
        them at test time with a hyperparam. alpha
        """
        if self.training:
            if self._mask_drop is None:
                _bs, c, _h, _w = x1.size()
                assert c == x2.size(1), 'Number of features is different'
                self._mask_drop = torch.ones_like(x1)
            mask_drop = (1 - alpha_rate) * F.dropout(self._mask_drop,
                alpha_rate)
            x1 = (x1 - alpha_rate * x2) / max(1e-08, 1 - alpha_rate)
            x = mask_drop * x1 + (1 - mask_drop) * x2
        else:
            x = (1 - alpha_rate) * x1 + alpha_rate * x2
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
