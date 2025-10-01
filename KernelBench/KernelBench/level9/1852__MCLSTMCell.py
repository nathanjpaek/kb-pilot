from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from typing import Tuple


class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: 'int', out_features: 'int'):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Perform forward pass through the normalised gate"""
        return torch.sigmoid(self.fc(x))


class _NormalizedGate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: 'int', out_shape: 'Tuple[int, int]',
        normalizer: 'str'):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape
            [0] * out_shape[1])
        self.out_shape = out_shape
        if normalizer == 'normalized_sigmoid':
            self.activation = nn.Sigmoid()
        elif normalizer == 'normalized_relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown normalizer {normalizer}. Must be one of {'normalized_sigmoid', 'normalized_relu'}"
                )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Perform forward pass through the normalized gate"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)


class _MCLSTMCell(nn.Module):
    """The logic of the MC-LSTM cell"""

    def __init__(self, mass_input_size: 'int', aux_input_size: 'int',
        hidden_size: 'int', cfg: 'Config'):
        super(_MCLSTMCell, self).__init__()
        self.cfg = cfg
        self._hidden_size = hidden_size
        gate_inputs = aux_input_size + hidden_size + mass_input_size
        self.output_gate = _Gate(in_features=gate_inputs, out_features=
            hidden_size)
        self.input_gate = _NormalizedGate(in_features=gate_inputs,
            out_shape=(mass_input_size, hidden_size), normalizer=
            'normalized_sigmoid')
        self.redistribution = _NormalizedGate(in_features=gate_inputs,
            out_shape=(hidden_size, hidden_size), normalizer='normalized_relu')
        self._reset_parameters()

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            nn.init.constant_(self.output_gate.fc.bias, val=self.cfg.
                initial_forget_bias)

    def forward(self, x_m: 'torch.Tensor', x_a: 'torch.Tensor') ->Tuple[
        torch.Tensor, torch.Tensor]:
        """Perform forward pass on the MC-LSTM cell.

        Parameters
        ----------
        x_m : torch.Tensor
            Mass input that will be conserved by the network.
        x_a : torch.Tensor
            Auxiliary inputs that will be used to modulate the gates but whose information won't be stored internally
            in the MC-LSTM cells.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Outgoing mass and memory cells per time step of shape [sequence length, batch size, hidden size]

        """
        _, batch_size, _ = x_m.size()
        ct = x_m.new_zeros((batch_size, self._hidden_size))
        m_out, c = [], []
        for xt_m, xt_a in zip(x_m, x_a):
            mt_out, ct = self._step(xt_m, xt_a, ct)
            m_out.append(mt_out)
            c.append(ct)
        m_out, c = torch.stack(m_out), torch.stack(c)
        return m_out, c

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        features = torch.cat([xt_m, xt_a, c / (c.norm(1) + 1e-05)], dim=-1)
        i = self.input_gate(features)
        r = self.redistribution(features)
        o = self.output_gate(features)
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        return o * m_new, (1 - o) * m_new


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'mass_input_size': 4, 'aux_input_size': 4, 'hidden_size': 
        4, 'cfg': _mock_config(initial_forget_bias=4)}]
