import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """利用卷积 + 最大池化得到句子嵌入"""

    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim
        =5, hidden_size=230):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3,
            padding=1)
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2)


def get_inputs():
    return [torch.rand([4, 60, 60])]


def get_init_inputs():
    return [[], {'max_length': 4}]
