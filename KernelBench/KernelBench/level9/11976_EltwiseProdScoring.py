import torch
import torch.nn as nn


class EltwiseProdScoring(nn.Module):
    """
    Linearly mapping h and v to the same dimension, and do a elementwise
    multiplication and a linear scoring
    """

    def __init__(self, h_dim, a_dim, dot_dim=256):
        """Initialize layer."""
        super(EltwiseProdScoring, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        """Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        """
        target = self.linear_in_h(h).unsqueeze(1)
        context = self.linear_in_a(all_u_t)
        eltprod = torch.mul(target, context)
        logits = self.linear_out(eltprod).squeeze(2)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'h_dim': 4, 'a_dim': 4}]
