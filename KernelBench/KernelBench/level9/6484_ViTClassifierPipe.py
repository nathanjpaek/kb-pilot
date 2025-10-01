from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class ViTClassifierPipe(nn.Module):

    def __init__(self, config: 'ViTConfig'):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels
            ) if config.num_labels > 0 else nn.Identity()

    def forward(self, args):
        hidden_states = args[0]
        hidden_states = self.layernorm(hidden_states)
        logits = self.classifier(hidden_states[:, 0, :])
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, layer_norm_eps=1,
        num_labels=4)}]
