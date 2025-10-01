import torch
import torch.nn as nn


class Normal_Model(nn.Module):
    """
    Example of a module for modeling a probability distribution. This is set up with all pieces
    required for use with the rest of this package. (initial parameters; as well as implimented
    constrain, forward, and log_prob methods)
    """

    def __init__(self, init_mean: 'torch.Tensor'=torch.Tensor([0]),
        init_std: 'torch.Tensor'=torch.Tensor([1])):
        super(Normal_Model, self).__init__()
        self.mean = nn.Parameter(init_mean, requires_grad=True)
        self.std = nn.Parameter(init_std, requires_grad=True)
        self.ln2p = nn.Parameter(torch.log(2 * torch.Tensor([torch.pi])),
            requires_grad=False)

    def constrain(self):
        """
        Method to run on "constrain" step of training. Easiest method for optimization under
        constraint is Projection Optimization by simply clamping parameters to bounds after each
        update. This is certainly not the most efficent way, but it gets the job done.
        """
        eps = 1e-06
        self.std.data = model.std.data.clamp(min=eps)

    def log_prob(self, x):
        """
        Returns the log probability of the items in tensor 'x' according to the probability
        distribution of the module.
        """
        return -torch.log(self.std.unsqueeze(-1)) - self.ln2p / 2 - ((x -
            self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)).pow(2) / 2

    def forward(self, x):
        """Returns the probability of the items in tensor 'x' according to the probability distribution of the module."""
        return self.log_prob(x).exp()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
