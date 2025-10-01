import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, dropout=0.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        """
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)
        score = self.dropout(F.softmax(score, dim=-1))
        return score.reshape((batch_size, num_of_timesteps, num_of_vertices,
            num_of_vertices))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
