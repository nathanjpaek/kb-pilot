import torch
from typing import Optional
import torch.nn.functional as F
from torch import nn
import torch.utils.data
import torch.nn
from torch.nn import RNNCellBase
import torch.multiprocessing
from torch.nn import Identity


class LayerNormGRUCell(RNNCellBase):
    """
    Implements GRUCell with layer normalisation and zone-out on top.
    It inherits the base RNN cell whose trainable weight matrices are used.

    References:
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." (2016).
    [2] Krueger, David, et al. "Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations." (2016).

    :param input_size: Number of input features to the cell
    :param hidden_size: Number of hidden states in the cell
    :param use_layer_norm: If set to True, layer normalisation is applied to
                           reset, update and new tensors before activation.
    :param dropout: Dropout probability for the hidden states [0,1]
    """

    def __init__(self, input_size: 'int', hidden_size: 'int',
        use_layer_norm: 'bool'=False, dropout: 'float'=0.0):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size,
            bias=False, num_chunks=3)
        self.dropout = dropout
        self.ln_r = nn.LayerNorm(self.hidden_size
            ) if use_layer_norm else Identity()
        self.ln_z = nn.LayerNorm(self.hidden_size
            ) if use_layer_norm else Identity()
        self.ln_n = nn.LayerNorm(self.hidden_size
            ) if use_layer_norm else Identity()

    def forward(self, input: 'torch.Tensor', hx: 'Optional[torch.Tensor]'=None
        ) ->torch.Tensor:
        if hx is None:
            hx = input.new_zeros(size=(input.size(0), self.hidden_size),
                requires_grad=False)
        ih = input.mm(self.weight_ih.t())
        hh = hx.mm(self.weight_hh.t())
        i_r, i_z, i_n = ih.chunk(3, dim=1)
        h_r, h_z, h_n = hh.chunk(3, dim=1)
        r = torch.sigmoid(self.ln_r(i_r + h_r))
        z = torch.sigmoid(self.ln_z(i_z + h_z))
        n = torch.tanh(self.ln_n(i_n + r * h_n))
        new_h = (torch.tensor(1.0) - z) * n + z * hx
        if self.dropout > 0.0:
            bernouli_mask = F.dropout(torch.ones_like(new_h), p=self.
                dropout, training=bool(self.training))
            new_h = bernouli_mask * new_h + (torch.tensor(1.0) - bernouli_mask
                ) * hx
        return new_h


class Identity(nn.Module):
    """
    Implements an identity torch module where input is passed as it is to output.
    There are no parameters in the module.
    """

    def __init__(self) ->None:
        super(Identity, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return input


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
