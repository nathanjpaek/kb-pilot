from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.utils.checkpoint


class BeitPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps) if config.use_mean_pooling else None

    def forward(self, hidden_states):
        if self.layernorm is not None:
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            pooled_output = hidden_states[:, 0]
        return pooled_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(use_mean_pooling=4, hidden_size=4,
        layer_norm_eps=1)}]
