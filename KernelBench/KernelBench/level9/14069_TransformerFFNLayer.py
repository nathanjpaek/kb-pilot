import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.utils.data
import torch.optim
import torch.distributions


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[
            module_name]
    return '{}.{}.{}'.format(module_name, module_instance._instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


class TransformerFFNLayer(nn.Module):

    def __init__(self, hidden_size, filter_size, padding='SAME',
        kernel_size=1, dropout=0.0, act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size,
                padding=kernel_size // 2)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(nn.ConstantPad1d((kernel_size - 1, 0
                ), 0.0), nn.Conv1d(hidden_size, filter_size, kernel_size))
        self.ffn_2 = Linear(filter_size, hidden_size)

    def forward(self, x, incremental_state=None):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                prev_input = saved_state['prev_input']
                x = torch.cat((prev_input, x), dim=0)
            x = x[-self.kernel_size:]
            saved_state['prev_input'] = x
            self._set_input_buffer(incremental_state, saved_state)
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5
        if incremental_state is not None:
            x = x[-1:]
        if self.act == 'gelu':
            x = F.gelu(x)
        if self.act == 'relu':
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, 'f') or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(self, incremental_state, 'f', buffer)

    def clear_buffer(self, incremental_state):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                del saved_state['prev_input']
            self._set_input_buffer(incremental_state, saved_state)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'filter_size': 4}]
