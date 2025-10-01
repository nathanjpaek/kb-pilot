import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from typing import Union
from typing import List
from typing import Tuple
from typing import Optional
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Dict
from torch.nn.modules.module import _IncompatibleKeys


def _compute_last_states(h_n: 'List[torch.Tensor]', c_n:
    'List[torch.Tensor]', seq_lengths: 'List[int]') ->Tuple[torch.Tensor,
    torch.Tensor]:
    """
    Given h and c values of all time steps, this function computes the h and c values for each sequence at their last timestep (this can vary across sequences with different sequence lengths).

    Args:
        h_n: A list of hidden state values across all timesteps.
        c_n: A list of cell state values across all timesteps.
        seq_lengths: the length parameter used in the torch.nn.utils.rnn.packed_padded_sequence function to create a PackedSequence. This can be computed using the _compute_seq_lengths function.

    Returns:
        h_last: Contains the last hidden state values for each of the sequences.
                If the i'th sequence has a length of l_i, then h_last[i,:] contains the hidden state corresponding to the i'th sequence at timestep l_i.
        c_last: The structure is the same as h_last, except that it contains the last cell state values for each of the sequences.
    """
    max_batch_size = len(seq_lengths)
    hidden_size = h_n[0].shape[-1]
    h_last = torch.zeros(max_batch_size, hidden_size)
    c_last = torch.zeros(max_batch_size, hidden_size)
    for i, seq_len in enumerate(seq_lengths):
        h_last[i, :] = h_n[seq_len - 1][i, :]
        c_last[i, :] = c_n[seq_len - 1][i, :]
    return h_last, c_last


def _compute_seq_lengths(batch_sizes: 'torch.Tensor') ->List[int]:
    """
    Computes the sequence lengths (the length parameter used in the packed_padded_sequence function to create a PackedSequence).

    Args:
        batch_sizes: Contains the batch sizes as stored in a PackedSequence

    Returns:
        running_seq_lengths: the length parameter used in the torch.nn.utils.rnn.packed_padded_sequence function to create a PackedSequence.
        It's a list of the same length as batch_sizes.
    """
    max_batch_size = batch_sizes[0]
    if len(batch_sizes) == 1:
        return [1] * max_batch_size
    running_seq = 0
    running_seq_lengths = []
    for i in range(1, len(batch_sizes)):
        delta = batch_sizes[i - 1].item() - batch_sizes[i].item()
        running_seq += 1
        running_seq_lengths += delta * [running_seq]
    running_seq += 1
    running_seq_lengths += batch_sizes[-1].item() * [running_seq]
    running_seq_lengths.reverse()
    return running_seq_lengths


def _concat_sequence_directions(forward:
    'Union[List[torch.Tensor], Tuple[torch.Tensor]]', reverse:
    'Union[List[torch.Tensor], Tuple[torch.Tensor]]', dim: 'int') ->Tuple[torch
    .Tensor]:
    """
    Given two list/tuple of same length containing tensors, this function returns a concatenation along dimension d. So, output[i] : concatenation of forward[i] and reverse[i] along dimension dim.
    forward[i] and reverse[i] should have the same shape. This function is used for concatenating the outputs of the forward and reverse layer of a bidirectional LSTM.

    Args:
        forward: list/tuple containing n tensors, representing the output of the forward layer.
        reverse: list/tuple containing n tensors, representing the output of the backward layer.
        dim: the dimension along which the sequence of tensors within forward and reverse will be concatenated.
    Returns:
        output: list/tuple containing n concatenated tensors.
    """
    if len(forward) != len(reverse):
        raise ValueError(
            'The forward and reverse layer output sequences should have the same length'
            )
    seq_length = len(forward)
    output = [0] * seq_length
    for i in range(seq_length):
        output[i] = torch.cat((forward[i], reverse[i]), dim=dim)
    return output


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

    def set_max_batch_length(self, max_batch_length: 'int') ->None:
        """
        Sets max batch length
        """
        self.ih.max_batch_len = max_batch_length
        self.hh.max_batch_len = max_batch_length

    def forward(self, x: 'torch.Tensor', h_prev: 'torch.Tensor', c_prev:
        'torch.Tensor', batch_size_t: 'Optional[int]'=None) ->Tuple[torch.
        Tensor, torch.Tensor]:
        if batch_size_t is None:
            gates = self.ih(x) + self.hh(h_prev)
        else:
            gates = self.ih(x) + self.hh(h_prev[:batch_size_t, :])
        i_t_input, f_t_input, g_t_input, o_t_input = torch.split(gates,
            self.hidden_size, 1)
        i_t = torch.sigmoid(i_t_input)
        f_t = torch.sigmoid(f_t_input)
        g_t = torch.tanh(g_t_input)
        o_t = torch.sigmoid(o_t_input)
        if batch_size_t is None:
            c_t = f_t * c_prev + i_t * g_t
        else:
            c_t = f_t * c_prev[:batch_size_t, :] + i_t * g_t
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

    def set_max_batch_length(self, max_batch_length: 'int') ->None:
        """
        Sets max batch length. Useful for PackedSequences
        """
        self.cell.set_max_batch_length(max_batch_length)

    def forward(self, x: 'Union[torch.Tensor, Tuple]', state_init:
        'Tuple[torch.Tensor, torch.Tensor]', batch_sizes:
        'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, Tuple[torch.
        Tensor, torch.Tensor]]:
        """
        Implements the forward pass of the DPLSTMLayer when a sequence is given in input.

        Args:
            x: Input sequence to the DPLSTMCell of shape ``[T, B, D]``.
            state_init: Initial state of the LSTMCell as a tuple ``(h_0, c_0)``
                where ``h_0`` is the initial hidden state and ``c_0`` is the
                initial cell state of the DPLSTMCell
            batch_sizes: Contains the batch sizes as stored in PackedSequence


        Returns:
            ``output, (h_n, c_n)`` where, ``output`` is of shape ``[T, B, H]`` and is a
            tensor containing the output features (``h_t``) from the last layer of the
            DPLSTMCell for each timestep ``t``. ``h_n`` is of shape ``[B, H]`` and is a
            tensor containing the hidden state for ``t = T``. ``c_n`` is of shape ``[B, H]``
            tensor containing the cell state for ``t = T``.
        """
        if batch_sizes is not None:
            seq_length = batch_sizes.size(0)
            if self.reverse:
                x = tuple(reversed(x))
                batch_sizes = batch_sizes.flip(0)
        else:
            seq_length, _batch_sz, _ = x.shape
            if self.reverse:
                x = x.flip(0)
            x = torch.unbind(x, dim=0)
        h_0, c_0 = state_init
        h_n = [h_0]
        c_n = [c_0]
        batch_size_prev = h_0.shape[0]
        for t in range(seq_length):
            if batch_sizes is not None:
                batch_size_t = batch_sizes[t].item()
                delta = batch_size_t - batch_size_prev
                if delta > 0:
                    h_cat = torch.cat((h_n[t], h_0[batch_size_prev:
                        batch_size_t, :]), 0)
                    c_cat = torch.cat((c_n[t], c_0[batch_size_prev:
                        batch_size_t, :]), 0)
                    h_next, c_next = self.cell(x[t], h_cat, c_cat, batch_size_t
                        )
                else:
                    h_next, c_next = self.cell(x[t], h_n[t], c_n[t],
                        batch_size_t)
            else:
                h_next, c_next = self.cell(x[t], h_n[t], c_n[t])
            if self.dropout:
                h_next = self.dropout_layer(h_next)
            h_n.append(h_next)
            c_n.append(c_next)
            batch_size_prev = h_next.shape[0]
        if batch_sizes is None:
            h_n = torch.stack(h_n[1:], dim=0)
            return h_n.flip(0) if self.reverse else h_n, (h_n[-1], c_n[-1])
        else:
            seq_lengths = _compute_seq_lengths(batch_sizes)
            h_temp, c_temp = h_n[1:], c_n[1:]
            h_last, c_last = _compute_last_states(h_temp, c_temp, seq_lengths)
            if self.reverse:
                h_temp = tuple(reversed(h_temp))
            return h_temp, (h_last, c_last)


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

    def set_max_batch_length(self, max_batch_length: 'int') ->None:
        """
        Sets max batch length
        """
        self.forward_layer.set_max_batch_length(max_batch_length)
        self.reverse_layer.set_max_batch_length(max_batch_length)

    def forward(self, x: 'torch.Tensor', state_init:
        'Tuple[torch.Tensor, torch.Tensor]', batch_sizes:
        'Optional[torch.Tensor]'=None) ->Tuple[torch.Tensor, Tuple[torch.
        Tensor, torch.Tensor]]:
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
            state_init: Initial state of the LSTM as a tuple ``(h_0, c_0)``, where
                ``h_0`` of shape ``[P, B, H]`` contains the initial hidden state, and
                ``c_0`` of shape ``[P, B, H]``  contains the initial cell state. This
                argument can be (and defaults to) None, in which case zero tensors
                will be used.

         Returns:
            ``output, (h_n, c_n)`` where, ``output`` is of shape ``[T, B, H * P]`` and is a
            tensor containing the output features (``h_t``) from the last layer of the
            DPLSTM for each timestep ``t``. ``h_n`` is of shape ``[P, B, H]`` and contains
            the hidden state for ``t = T``. ``c_n`` is of shape ``[P, B, H]`` and contains
            the cell state for ``t = T``.
        """
        h0, c0 = state_init
        h0_f, h0_r = h0.unbind(0)
        c0_f, c0_r = c0.unbind(0)
        out_f, (h_f, c_f) = self.forward_layer(x, (h0_f, c0_f), batch_sizes)
        out_r, (h_r, c_r) = self.reverse_layer(x, (h0_r, c0_r), batch_sizes)
        if batch_sizes is None:
            out = torch.cat([out_f, out_r], dim=-1)
        else:
            out = _concat_sequence_directions(out_f, out_r, -1)
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

    def forward(self, x: 'Union[torch.Tensor, PackedSequence]', state_init:
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
            x: Input sequence to the DPLSTM of shape ``[T, B, D]``. Or it can be a PackedSequence.
            state_init: Initial state of the LSTM as a tuple ``(h_0, c_0)``, where:
                - ``h_0`` of shape ``[L*P, B, H]`` contains the initial hidden state
                - ``c_0`` of shape ``[L*P, B, H]`` contains the initial cell state

                This argument can be (and defaults to) None, in which case zero tensors will be used.

         Returns:
            ``output, (h_n, c_n)`` where, ``output`` is of shape ``[T, B, H * P]`` and is a
            tensor containing the output features (``h_t``) from the last layer of the DPLSTM
            for each timestep ``t``. ``h_n`` is of shape ``[L * P, B, H]`` and contains the
            hidden state for ``t = T``. ``c_n`` is of shape ``[L * P, B, H]`` and contains
            the cell state for ``t = T``.
        """
        if isinstance(x, PackedSequence):
            x, batch_sizes, sorted_indices, unsorted_indices = x
            B = batch_sizes[0].item()
            _, _D = x.shape
            x = x.split(tuple(batch_sizes))
            for layer in self.layers:
                layer.set_max_batch_length(B)
        else:
            sorted_indices = None
            unsorted_indices = None
            batch_sizes = None
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
            h_0s = self._permute_hidden(h_0s, sorted_indices, 2)
        if c_0s is None:
            c_0s = torch.zeros(L, P, B, self.hidden_size, dtype=x[0].dtype,
                device=x[0].device)
        else:
            c_0s = c_0s.reshape([L, P, B, H])
            c_0s = self._permute_hidden(c_0s, sorted_indices, 2)
        hs: 'List[torch.Tensor]' = []
        cs: 'List[torch.Tensor]' = []
        for layer, h0, c0 in zip(self.layers, h_0s, c_0s):
            if not self.bidirectional:
                h0 = h0.squeeze(0)
                c0 = c0.squeeze(0)
            x, (h, c) = layer(x, (h0, c0), batch_sizes)
            if not self.bidirectional:
                h = h.unsqueeze(0)
                c = c.unsqueeze(0)
            hs.append(h)
            cs.append(c)
        hs = torch.cat(hs, dim=0)
        cs = torch.cat(cs, dim=0)
        if batch_sizes is not None:
            seq_lengths = _compute_seq_lengths(batch_sizes)
            packed_data = pack_padded_sequence(pad_sequence(x, batch_first=
                False), seq_lengths, batch_first=True)[0]
            out = PackedSequence(packed_data, batch_sizes, sorted_indices,
                unsorted_indices)
        else:
            out = self._rearrange_batch_dim(x)
        return out, (self._permute_hidden(hs, unsorted_indices), self.
            _permute_hidden(cs, unsorted_indices))

    def _permute_hidden(self, x: 'torch.Tensor', permutation:
        'Optional[torch.Tensor]'=None, dim: 'int'=1) ->torch.Tensor:
        if permutation is None:
            return x
        if dim == 1:
            return x[:, permutation, :]
        elif dim == 2:
            return x[:, :, permutation, :]

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
