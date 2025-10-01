from _paritybench_helpers import _mock_config
import torch
from torch import nn


class GPT2Postprocessing(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
            bias=False)

    def forward(self, hidden_states):
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, layer_norm_epsilon=1,
        vocab_size=4)}]
