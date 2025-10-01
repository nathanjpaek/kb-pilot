from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for classification tasks
        Layers:
        - dropout
        - dense layer (default xlsr hidden size = 1024)
        - relu
        - dropout
        - classificiation layer of size num_labels
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.hidden_dropout(x)
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_dropout=0.5, hidden_size=4,
        final_dropout=0.5, num_labels=4)}]
