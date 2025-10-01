from _paritybench_helpers import _mock_config
import math
import torch
from torch import nn


def gelu(x):
    """Gaussian Error Linear Unitという活性化関数です。
    LeLUが0でカクっと不連続なので、そこを連続になるように滑らかにした形のLeLUです。
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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


class BertPredictionHeadTransform(nn.Module):
    """MaskedWordPredictionsにて、BERTからの特徴量を変換するモジュール（入出力のサイズは同じ）"""

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        """hidden_statesはsequence_output:[minibatch, seq_len, hidden_size]"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
