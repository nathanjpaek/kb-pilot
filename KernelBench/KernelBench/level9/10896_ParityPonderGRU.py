from torch.nn import Module
import torch
from torch import nn
from typing import Tuple
import torch.utils.data
import torch.nn.functional
import torch.autograd


class ParityPonderGRU(Module):
    """
    ## PonderNet with GRU for Parity Task

    This is a simple model that uses a [GRU Cell](https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)
    as the step function.

    This model is for the [Parity Task](../parity.html) where the input is a vector of `n_elems`.
    Each element of the vector is either `0`, `1` or `-1` and the output is the parity
    - a binary value that is true if the number of `1`s is odd and false otherwise.

    The prediction of the model is the log probability of the parity being $1$.
    """

    def __init__(self, n_elems: 'int', n_hidden: 'int', max_steps: 'int'):
        """
        * `n_elems` is the number of elements in the input vector
        * `n_hidden` is the state vector size of the GRU
        * `max_steps` is the maximum number of steps $N$
        """
        super().__init__()
        self.max_steps = max_steps
        self.n_hidden = n_hidden
        self.gru = nn.GRUCell(n_elems, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)
        self.lambda_layer = nn.Linear(n_hidden, 1)
        self.lambda_prob = nn.Sigmoid()
        self.is_halt = False

    def forward(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor]:
        """
        * `x` is the input of shape `[batch_size, n_elems]`

        This outputs a tuple of four tensors:

        1. $p_1 \\dots p_N$ in a tensor of shape `[N, batch_size]`
        2. $\\hat{y}_1 \\dots \\hat{y}_N$ in a tensor of shape `[N, batch_size]` - the log probabilities of the parity being $1$
        3. $p_m$ of shape `[batch_size]`
        4. $\\hat{y}_m$ of shape `[batch_size]` where the computation was halted at step $m$
        """
        batch_size = x.shape[0]
        h = x.new_zeros((x.shape[0], self.n_hidden))
        h = self.gru(x, h)
        p = []
        y = []
        un_halted_prob = h.new_ones((batch_size,))
        halted = h.new_zeros((batch_size,))
        p_m = h.new_zeros((batch_size,))
        y_m = h.new_zeros((batch_size,))
        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = h.new_ones(h.shape[0])
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]
            y_n = self.output_layer(h)[:, 0]
            p_n = un_halted_prob * lambda_n
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            halt = torch.bernoulli(lambda_n) * (1 - halted)
            p.append(p_n)
            y.append(y_n)
            p_m = p_m * (1 - halt) + p_n * halt
            y_m = y_m * (1 - halt) + y_n * halt
            halted = halted + halt
            h = self.gru(x, h)
            if self.is_halt and halted.sum() == batch_size:
                break
        return torch.stack(p), torch.stack(y), p_m, y_m


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_elems': 4, 'n_hidden': 4, 'max_steps': 4}]
