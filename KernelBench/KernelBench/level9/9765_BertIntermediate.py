from _paritybench_helpers import _mock_config
import math
import torch
from torch import nn


def gelu(x):
    """Gaussian Error Linear Unitという活性化関数です。
    LeLUが0でカクっと不連続なので、そこを連続になるように滑らかにした形のLeLUです。
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    """BERTのTransformerBlockモジュールのFeedForwardです"""

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        """
        hidden_states： BertAttentionの出力テンソル
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, intermediate_size=4)}]
