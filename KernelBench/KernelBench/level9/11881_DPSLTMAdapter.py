import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from typing import Tuple
from typing import List
from typing import Optional
from typing import Dict
from typing import Union
from torch.nn.modules.module import _IncompatibleKeys


def filter_out_old_keys(self, state_dict, prefix, local_metadata):
    new_state_dict = {param_name: param_value for param_name, param_value in
        state_dict.items() if param_name not in self.old_to_new}
    return new_state_dict


class LSTMLinear(nn.Linear):
    """
    This function is the same as a nn.Linear layer, except that in the backward pass
    the grad_samples get accumulated (instead of being concatenated as in the standard
    nn.Linear)
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias:
        'bool'=True):
        super().__init__(in_features, out_features, bias)


class DPLSTMCell(nn.Module):
    """
    Internal-only class. Implements *one* step of LSTM so that a LSTM layer can be seen as repeated
    applications of this class.
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.ih = LSTMLinear(input_size, 4 * hidden_size, bias=self.bias)
        self.hh = LSTMLinear(hidden_size, 4 * hidden_size, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets parameters by initializing them from an uniform distribution.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: 'torch.Tensor', h_prev: 'torch.Tensor', c_prev:
        'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        gates = self.ih(x) + self.hh(h_prev)
        i_t_input, f_t_input, g_t_input, o_t_input = torch.split(gates,
            self.hidden_size, 1)
        i_t = torch.sigmoid(i_t_input)
        f_t = torch.sigmoid(f_t_input)
        g_t = torch.tanh(g_t_input)
        o_t = torch.sigmoid(o_t_input)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class DPLSTMLayer(nn.Module):
    """
    Implements *one* layer of LSTM in a way amenable to differential privacy.
    We don't expect you to use this directly: use DPLSTM instead :)
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool',
        dropout: 'float', reverse: 'bool'=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.reverse = reverse
        self.cell = DPLSTMCell(input_size=input_size, hidden_size=
            hidden_size, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: 'torch.Tensor', state_init:
        'Tuple[torch.Tensor, torch.Tensor]') ->Tuple[torch.Tensor, Tuple[
        torch.Tensor, torch.Tensor]]:
        """
        Implements the forward pass of the DPLSTMLayer when a sequence is given in input.

        Args:
            x: Input sequence to the DPLSTMCell of shape ``[T, B, D]``.
            state_init: Initial state of the LSTMCell as a tuple ``(h_0, c_0)``
                where ``h_0`` is the initial hidden state and ``c_0`` is the
                initial cell state of the DPLSTMCell

        Returns:
            ``output, (h_n, c_n)`` where:
            - ``output`` is of shape ``[T, B, H]`` and is a tensor containing the output
                features (``h_t``) from the last layer of the DPLSTMCell for each timestep ``t``.
            - ``h_n`` is of shape ``[B, H]`` and is a tensor containing the hidden state for ``t = T``.
            - ``c_n`` is of shape ``[B, H]`` tensor containing the cell state for ``t = T``.
        """
        seq_length, _batch_sz, _ = x.shape
        if self.reverse:
            x = x.flip(0)
        x = torch.unbind(x, dim=0)
        h_0, c_0 = state_init
        h_n = [h_0]
        c_n = [c_0]
        for t in range(seq_length):
            h_next, c_next = self.cell(x[t], h_n[t], c_n[t])
            if self.dropout:
                h_next = self.dropout_layer(h_next)
            h_n.append(h_next)
            c_n.append(c_next)
        h_n = torch.stack(h_n[1:], dim=0)
        return h_n.flip(0) if self.reverse else h_n, (h_n[-1], c_n[-1])


class BidirectionalDPLSTMLayer(nn.Module):
    """
    Implements *one* layer of Bidirectional LSTM in a way amenable to differential privacy.
    We don't expect you to use this directly: use DPLSTM instead :)
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', bias: 'bool',
        dropout: 'float'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.forward_layer = DPLSTMLayer(input_size=input_size, hidden_size
            =hidden_size, bias=bias, dropout=dropout, reverse=False)
        self.reverse_layer = DPLSTMLayer(input_size=input_size, hidden_size
            =hidden_size, bias=bias, dropout=dropout, reverse=True)

    def forward(self, x: 'torch.Tensor', state_init:
        'Tuple[torch.Tensor, torch.Tensor]') ->Tuple[torch.Tensor, Tuple[
        torch.Tensor, torch.Tensor]]:
        """
        Implements the forward pass of the DPLSTM when a sequence is input.

        Dimensions as follows:
            - B: Batch size
            - T: Sequence length
            - D: LSTM input hidden size (eg from a word embedding)
            - H: LSTM output hidden size
            - P: number of directions (2 if bidirectional, else 1)

        Args:
            x: Input sequence to the DPLSTM of shape ``[T, B, D]``
            state_init: Initial state of the LSTM as a tuple ``(h_0, c_0)``, where:
                    - h_0 of shape ``[P, B, H]  contains the initial hidden state
                    - c_0 of shape ``[P, B, H]  contains the initial cell state
                    This argument can be (and defaults to) None, in which case zero tensors will be used.

         Returns:
            ``output, (h_n, c_n)`` where:
            - ``output`` is of shape ``[T, B, H * P]`` and is a tensor containing the output
                features (``h_t``) from the last layer of the DPLSTM for each timestep ``t``.
            - ``h_n`` is of shape ``[P, B, H]`` and contains the hidden state for ``t = T``.
            - ``c_n`` is of shape ``[P, B, H]`` and contains the cell state for ``t = T``.
        """
        h0, c0 = state_init
        h0_f, h0_r = h0.unbind(0)
        c0_f, c0_r = c0.unbind(0)
        out_f, (h_f, c_f) = self.forward_layer(x, (h0_f, c0_f))
        out_r, (h_r, c_r) = self.reverse_layer(x, (h0_r, c0_r))
        out = torch.cat([out_f, out_r], dim=-1)
        h = torch.stack([h_f, h_r], dim=0)
        c = torch.stack([c_f, c_r], dim=0)
        return out, (h, c)


class ParamRenamedModule(nn.Module):
    """
    This class defines a nn.Module whose parameters are renamed. This is useful when you want to
    reimplement a layer but make sure its state_dict and list of parameters are exactly the same
    as another reference layer so that you can have a drop-in replacement that does not depend on
    how your layer is actually implemented. In Opacus, this is used for DPLSTM, where our
    implementation leverages submodules and requires alignment to the state_dict of nn.LSTM.
    """

    def __init__(self, rename_map: 'Dict[str, str]'):
        """
        Initializes internal state. Subclass this instead of ``torch.nn.Module`` whenever you need
        to rename your model's state.

        Args:
            rename_map: mapping from old name -> new name for each parameter you want renamed.
                Note that this must be a 1:1 mapping!
        """
        super().__init__()
        self.old_to_new = rename_map
        self.new_to_old = {v: k for k, v in rename_map.items()}
        self._register_state_dict_hook(filter_out_old_keys)

    def _register_renamed_parameters(self):
        """
        Internal function. This function simply registers parameters under their new name. They will
        automatically mask their duplicates coming from submodules. This trick works because
        self.parameters() proceeds recursively from the top, going into submodules after processing
        items at the current level, and will not return duplicates.
        """
        for old_name, param in super().named_parameters():
            if old_name in self.old_to_new:
                new_name = self.old_to_new[old_name]
                self.register_parameter(new_name, param)

    def __setattr__(self, name: 'str', value: 'Union[Tensor, nn.Module]'
        ) ->None:
        """
        Whenever you set an attribute, eg `self.linear`, this is called to actually register it in
        any nn.Module. We rely on the masking trick explained in the docs for
        ``_register_renamed_parameters`` to make sure we replace things only once. If a new parameter
        in the rename list is detected, we rename and mask it so next time this is called we will
        no longer find it.
        """
        super().__setattr__(name, value)
        try:
            self._register_renamed_parameters()
        except AttributeError:
            pass

    def load_state_dict(self, state_dict: 'Dict[str, Tensor]', strict:
        'bool'=True):
        """
        Identical to ``torch.nn.Module.load_state_dict()`` but handles the renamed keys.
        """
        missing_keys, unexpected_keys = super().load_state_dict(state_dict,
            strict=False)
        missing_keys = [k for k in missing_keys if k not in self.old_to_new]
        if strict:
            error_msgs = []
            if len(unexpected_keys) > 0:
                error_msgs.insert(0,
                    'Unexpected key(s) in state_dict: {}. '.format(', '.
                    join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.
                    format(', '.join('"{}"'.format(k) for k in missing_keys)))
            if len(error_msgs) > 0:
                raise RuntimeError(
                    'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    self.__class__.__name__, '\n\t'.join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)


class DPLSTM(ParamRenamedModule):
    """
    DP-friendly drop-in replacement of the ``torch.nn.LSTM`` module.

    Its state_dict matches that of nn.LSTM exactly, so that after training it can be exported
    and loaded by an nn.LSTM for inference.

    Refer to nn.LSTM's documentation for all parameters and inputs.
    """

    def __init__(self, input_size: 'int', hidden_size: 'int', num_layers:
        'int'=1, bias: 'bool'=True, batch_first: 'bool'=False, dropout:
        'float'=0, bidirectional: 'bool'=False):
        rename_dict = self._make_rename_dict(num_layers, bias, bidirectional)
        super().__init__(rename_dict)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        LayerClass = BidirectionalDPLSTMLayer if bidirectional else DPLSTMLayer
        self.layers = nn.ModuleList([LayerClass(input_size=self.input_size if
            i == 0 else self.hidden_size * self.num_directions, hidden_size
            =self.hidden_size, bias=self.bias, dropout=self.dropout if i < 
            self.num_layers - 1 else 0) for i in range(num_layers)])

    def forward(self, x: 'torch.Tensor', state_init:
        'Optional[Tuple[torch.Tensor, torch.Tensor]]'=None) ->Tuple[torch.
        Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implements the forward pass of the DPLSTM when a sequence is input.

        Dimensions as follows:
            - B: Batch size
            - T: Sequence length
            - D: LSTM input hidden size (eg from a word embedding)
            - H: LSTM output hidden size
            - L: number of layers in the LSTM
            - P: number of directions (2 if bidirectional, else 1)

        Args:
            x: Input sequence to the DPLSTM of shape ``[T, B, D]``
            state_init: Initial state of the LSTM as a tuple ``(h_0, c_0)``, where:
                    - h_0 of shape ``[L*P, B, H]  contains the initial hidden state
                    - c_0 of shape ``[L*P, B, H]  contains the initial cell state
                    This argument can be (and defaults to) None, in which case zero tensors will be used.

         Returns:
            ``output, (h_n, c_n)`` where:
            - ``output`` is of shape ``[T, B, H * P]`` and is a tensor containing the output
                features (``h_t``) from the last layer of the DPLSTM for each timestep ``t``.
            - ``h_n`` is of shape ``[L * P, B, H]`` and contains the hidden state for ``t = T``.
            - ``c_n`` is of shape ``[L * P, B, H]`` and contains the cell state for ``t = T``.
        """
        x = self._rearrange_batch_dim(x)
        _T, B, _D = x.shape
        L = self.num_layers
        P = 2 if self.bidirectional else 1
        H = self.hidden_size
        h_0s, c_0s = state_init or (None, None)
        if h_0s is None:
            h_0s = torch.zeros(L, P, B, self.hidden_size, dtype=x[0].dtype,
                device=x[0].device)
        else:
            h_0s = h_0s.reshape([L, P, B, H])
        if c_0s is None:
            c_0s = torch.zeros(L, P, B, self.hidden_size, dtype=x[0].dtype,
                device=x[0].device)
        else:
            c_0s = c_0s.reshape([L, P, B, H])
        hs: 'List[torch.Tensor]' = []
        cs: 'List[torch.Tensor]' = []
        for layer, h0, c0 in zip(self.layers, h_0s, c_0s):
            if not self.bidirectional:
                h0 = h0.squeeze(0)
                c0 = c0.squeeze(0)
            x, (h, c) = layer(x, (h0, c0))
            if not self.bidirectional:
                h = h.unsqueeze(0)
                c = c.unsqueeze(0)
            hs.append(h)
            cs.append(c)
        hs = torch.cat(hs, dim=0)
        cs = torch.cat(cs, dim=0)
        out = self._rearrange_batch_dim(x)
        return out, (hs, cs)

    def _rearrange_batch_dim(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.batch_first:
            x = x.transpose(0, 1)
        return x

    def __repr__(self):
        s = f'DPLSTM({self.input_size}, {self.hidden_size}, bias={self.bias}'
        if self.batch_first:
            s += f', batch_first={self.batch_first}'
        if self.num_layers > 1:
            s += f', num_layers={self.num_layers}'
        if self.dropout:
            s += f', dropout={self.dropout}'
        if self.bidirectional:
            s += f', bidirectional={self.bidirectional}'
        return s

    def _make_rename_dict(self, num_layers, bias, bidirectional):
        """
        Programmatically constructs a dictionary old_name -> new_name to align with the param
        names used in ``torch.nn.LSTM``.
        """
        d = {}
        components = ['weight'] + ['bias' if bias else []]
        matrices = ['ih', 'hh']
        for i in range(num_layers):
            for c in components:
                for m in matrices:
                    nn_name = f'{c}_{m}_l{i}'
                    if bidirectional:
                        d[f'layers.{i}.forward_layer.cell.{m}.{c}'] = nn_name
                        d[f'layers.{i}.reverse_layer.cell.{m}.{c}'
                            ] = nn_name + '_reverse'
                    else:
                        d[f'layers.{i}.cell.{m}.{c}'] = nn_name
        return d


class DPSLTMAdapter(nn.Module):
    """
    Adapter for DPLSTM.
    LSTM returns a tuple, but our testing tools need the model to return a single tensor in output.
    We do this adaption here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dplstm = DPLSTM(*args, **kwargs)

    def forward(self, x):
        out, _rest = self.dplstm(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
