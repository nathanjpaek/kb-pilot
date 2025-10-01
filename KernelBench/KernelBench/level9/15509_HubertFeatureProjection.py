from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.utils.checkpoint


class HubertFeatureProjection(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.
            layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(conv_dim=[4, 4], layer_norm_eps=1,
        hidden_size=4, feat_proj_dropout=0.5)}]
