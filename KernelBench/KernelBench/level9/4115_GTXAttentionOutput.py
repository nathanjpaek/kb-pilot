from _paritybench_helpers import _mock_config
import torch
from torch import nn


class GTXAttentionOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, KnowMix_indices=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if KnowMix_indices is None:
            hidden_states = input_tensor + hidden_states
        else:
            if isinstance(KnowMix_indices, int):
                input_tensor[:, KnowMix_indices] = input_tensor[:,
                    KnowMix_indices] + hidden_states.squeeze(1)
            else:
                input_tensor[KnowMix_indices, :] = input_tensor[
                    KnowMix_indices, :] + hidden_states.squeeze(1)
            hidden_states = input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=
        0.5)}]
