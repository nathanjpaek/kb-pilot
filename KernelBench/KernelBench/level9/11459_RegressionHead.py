import abc
import torch
import torch.nn as nn
import torch.utils.data.dataset


class BaseHead(nn.Module, metaclass=abc.ABCMeta):
    pass


class RegressionHead(BaseHead):

    def __init__(self, hidden_size, hidden_dropout_prob):
        """From RobertaClassificationHead"""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, pooled):
        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        scores = self.out_proj(x)
        return scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'hidden_dropout_prob': 0.5}]
