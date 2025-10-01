from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.utils.checkpoint


class BertSelfOutput(nn.Module):

    def __init__(self, config, twin=False, merge=False):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if twin:
            self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if merge:
            self.act = ACT2FN[config.hidden_act]
            self.merge_layer = nn.Linear(config.hidden_size * 2, config.
                hidden_size)
            self.merge = True
        else:
            self.merge = False

    def forward(self, hidden_states, input_tensor):
        if type(hidden_states) == list:
            hidden_states0 = self.dense0(hidden_states[0])
            hidden_states1 = self.dense1(hidden_states[1])
            if self.merge:
                hidden_states = self.merge_layer(torch.cat([hidden_states0,
                    hidden_states1], dim=-1))
            else:
                hidden_states = (hidden_states0 + hidden_states1) / 2
        else:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, layer_norm_eps=1,
        hidden_dropout_prob=0.5)}]
