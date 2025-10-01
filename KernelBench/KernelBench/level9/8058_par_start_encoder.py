import torch
import numpy as np
from torch import nn


class par_start_encoder(nn.Module):
    """A network which makes the initial states a parameter of the network"""

    def __init__(self, nx, nsamples):
        super(par_start_encoder, self).__init__()
        self.start_state = nn.parameter.Parameter(data=torch.as_tensor(np.
            random.normal(scale=0.1, size=(nsamples, nx)), dtype=torch.float32)
            )

    def forward(self, ids):
        return self.start_state[ids]


def get_inputs():
    return [torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'nx': 4, 'nsamples': 4}]
