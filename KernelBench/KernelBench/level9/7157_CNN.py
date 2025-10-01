from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class CNN(nn.Module):
    """ CNN for heat shock protein classification """

    def __init__(self, model_cfg, in_channels, dropout_rate):
        super(CNN, self).__init__()
        self.embedder = model_cfg.embedder
        if self.embedder != 'OneHot':
            self.embed = nn.Linear(in_channels, model_cfg.embed_dim)
            in_channels = model_cfg.embed_dim
        self.conv = nn.Conv1d(in_channels, model_cfg.num_channels,
            model_cfg.kernel_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate if dropout_rate is not
            None else 0)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(model_cfg.num_channels, 7)

    def forward(self, x):
        if self.embedder != 'OneHot':
            x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = self.max_pool(x).reshape(len(x), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'model_cfg': _mock_config(embedder=4, embed_dim=4,
        num_channels=4, kernel_size=4), 'in_channels': 4, 'dropout_rate': 0.5}]
