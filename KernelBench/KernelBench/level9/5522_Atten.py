from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.nn.functional as F


class Atten(nn.Module):

    def __init__(self, config):
        super(Atten, self).__init__()
        hidden_size = config.hidden_size
        classifier_dropout = (config.classifier_dropout if config.
            classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels
        self.fc = nn.Linear(hidden_size, num_classes)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        x = self.dropout(x)
        M = self.tanh(x)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = x * alpha
        out = torch.sum(out, 1)
        out = self.fc(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, classifier_dropout=
        0.5, num_labels=4)}]
