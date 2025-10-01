from _paritybench_helpers import _mock_config
import torch
from torch import nn


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """LayerNormalization層です。
        学習済みモデルをそのままロードするため、学習済みモデルの変数名に変えています。
        オリジナルのGitHubの実装から変数名を変えています。
        weight→gamma、bias→beta
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BertOutput(nn.Module):
    """BERTのTransformerBlockモジュールのFeedForwardです"""

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states： BertIntermediateの出力テンソル
        input_tensor：BertAttentionの出力テンソル
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(intermediate_size=4, hidden_size=4,
        hidden_dropout_prob=0.5)}]
