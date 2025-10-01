import torch
import torch.nn as nn
from torch import logsumexp as logsumexp
import torch.nn.functional as F


class BalancedNet(nn.Module):
    """A torch.model used as a component of the HEMM module to determine the outcome as a function of confounders. 
    The balanced net consists of two different neural networks for the outcome and counteractual.
    """

    def __init__(self, D_in, H, D_out):
        """Instantiate two nn.Linear modules and assign them as member variables.

        Args:
            D_in: input dimension
            H: dimension of hidden layer
            D_out: output dimension
        """
        super(BalancedNet, self).__init__()
        self.f1 = nn.Linear(D_in, H)
        self.f2 = nn.Linear(H, D_out)
        self.cf1 = nn.Linear(D_in, H)
        self.cf2 = nn.Linear(H, D_out)

    def forward(self, x):
        """Accept a Variable of input data and return a Variable of output data.

        We can use Modules defined in the constructor as well as arbitrary operators on Variables.
        """
        h_relu = F.elu(self.f1(x))
        f = self.f2(h_relu)
        h_relu = F.elu(self.cf1(x))
        cf = self.cf2(h_relu)
        out = torch.cat((f, cf), dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4, 'H': 4, 'D_out': 4}]
