import torch
import torch.nn as nn


class EmbeddingModule(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(EmbeddingModule, self).__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_1 = nn.Conv1d(input_dim, output_dim, 1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv1d(output_dim, output_dim, 1)

    def forward(self, x):
        x = self.dropout(x.unsqueeze(3)).squeeze(3)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'dropout_rate': 0.5}]
