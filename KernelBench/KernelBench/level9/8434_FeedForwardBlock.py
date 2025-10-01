from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):

    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.d_model, config.d_ff)
        self.w_2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class FeedForwardBlock(nn.Module):

    def __init__(self, config):
        super(FeedForwardBlock, self).__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.feed_forward = PositionwiseFeedForward(config)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        x_ = self.norm(x)
        x_ = self.feed_forward(x_)
        return self.dropout(x_) + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(d_model=4, d_ff=4, dropout=0.5)}]
