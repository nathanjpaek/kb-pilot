from _paritybench_helpers import _mock_config
import math
import torch
from torch import nn


class AttnBertPooler(nn.Module):

    def __init__(self, config):
        super(AttnBertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.activation = nn.Tanh()
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0].view(len(hidden_states), -1, 1
            )
        scores = torch.matmul(hidden_states[:, 1:], first_token_tensor
            ) / math.sqrt(self.hidden_size)
        attn_token_tensor = torch.matmul(hidden_states[:, 1:].view(
            hidden_states.size(0), self.hidden_size, -1), scores)
        attn_token_tensor = attn_token_tensor.view(attn_token_tensor.size(0
            ), self.hidden_size)
        first_token_tensor = first_token_tensor.squeeze(2)
        pooled_token_tensor = torch.cat((attn_token_tensor,
            first_token_tensor), dim=-1)
        pooled_output = self.dense(pooled_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
