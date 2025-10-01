from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class LSTMCell(Module):
    """
    ## Long Short-Term Memory Cell

    LSTM Cell computes $c$, and $h$. $c$ is like the long-term memory,
    and $h$ is like the short term memory.
    We use the input $x$ and $h$ to update the long term memory.
    In the update, some features of $c$ are cleared with a forget gate $f$,
    and some features $i$ are added through a gate $g$.

    The new short term memory is the $	anh$ of the long-term memory
    multiplied by the output gate $o$.

    Note that the cell doesn't look at long term memory $c$ when doing the update. It only modifies it.
    Also $c$ never goes through a linear transformation.
    This is what solves vanishing and exploding gradients.

    Here's the update rule.

    egin{align}
    c_t &= \\sigma(f_t) \\odot c_{t-1} + \\sigma(i_t) \\odot 	anh(g_t) \\
    h_t &= \\sigma(o_t) \\odot 	anh(c_t)
    \\end{align}

    $\\odot$ stands for element-wise multiplication.

    Intermediate values and gates are computed as linear transformations of the hidden
    state and input.

    egin{align}
    i_t &= lin_x^i(x_t) + lin_h^i(h_{t-1}) \\
    f_t &= lin_x^f(x_t) + lin_h^f(h_{t-1}) \\
    g_t &= lin_x^g(x_t) + lin_h^g(h_{t-1}) \\
    o_t &= lin_x^o(x_t) + lin_h^o(h_{t-1})
    \\end{align}

    """

    def __init__(self, input_size: 'int', hidden_size: 'int', layer_norm:
        'bool'=False):
        super().__init__()
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for
                _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: 'torch.Tensor', h: 'torch.Tensor', c: 'torch.Tensor'):
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        ifgo = ifgo.chunk(4, dim=-1)
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]
        i, f, g, o = ifgo
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
        return h_next, c_next


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
