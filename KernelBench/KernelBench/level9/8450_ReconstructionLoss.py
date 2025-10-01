import torch
from functools import reduce
import torch.nn as nn


class BaseModule(nn.Module):
    """
    Implements the basic module.
    All other modules inherit from this one
    """

    def load_w(self, checkpoint_path):
        """
        Loads a checkpoint into the state_dict.

        :param checkpoint_path: the checkpoint file to be loaded.
        """
        device = torch.device('cuda:' + '1')
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))

    def __repr__(self):
        """
        String representation
        """
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)
        return good_old + '\n' + addition

    def __call__(self, *args, **kwargs):
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        """
        Number of parameters of the model.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)


class ReconstructionLoss(BaseModule):
    """
    Implements the reconstruction loss.
    """

    def __init__(self):
        """
        Class constructor.
        """
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, x_r):
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :return: the mean reconstruction loss (averaged along the batch axis).
        """
        L = torch.pow(x - x_r, 2)
        while L.dim() > 1:
            L = torch.sum(L, dim=-1)
        return torch.mean(L)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
