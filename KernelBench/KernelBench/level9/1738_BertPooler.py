from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.onnx


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.cls_position = config.mask_tokens

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.cls_position]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_inputs():
    return [torch.rand([4, 5, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, mask_tokens=4)}]
