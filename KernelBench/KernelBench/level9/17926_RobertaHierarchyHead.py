from _paritybench_helpers import _mock_config
import torch
from torch import nn


class RobertaHierarchyHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(RobertaHierarchyHead, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.i2h = nn.Linear(config.hidden_size + config.hidden_size,
            config.hidden_size)
        self.i2o = nn.Linear(config.hidden_size + config.hidden_size, self.
            num_labels)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = self.dropout(combined)
        combined = torch.tanh(combined)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=
        0.5), 'num_labels': 4}]
