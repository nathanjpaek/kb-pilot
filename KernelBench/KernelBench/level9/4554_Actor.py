import torch
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor model

        Parameters:
              args (object): Parameter class
    """

    def __init__(self, state_dim, action_dim, wwid):
        super(Actor, self).__init__()
        self.wwid = torch.Tensor([wwid])
        l1 = 400
        l2 = 300
        self.f1 = nn.Linear(state_dim, l1)
        self.ln1 = nn.LayerNorm(l1)
        self.f2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)
        self.w_out = nn.Linear(l2, action_dim)

    def forward(self, input):
        """Method to forward propagate through the actor's graph

            Parameters:
                  input (tensor): states

            Returns:
                  action (tensor): actions


        """
        out = F.elu(self.f1(input))
        out = self.ln1(out)
        out = F.elu(self.f2(out))
        out = self.ln2(out)
        return torch.sigmoid(self.w_out(out))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'wwid': 4}]
