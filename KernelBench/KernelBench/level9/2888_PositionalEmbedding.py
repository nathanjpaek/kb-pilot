import torch
import numpy as np
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, embed_dim, max_seq_len):
        super(PositionalEmbedding, self).__init__()
        position_encodings = np.array([[(pos / np.power(10000, 2.0 * (i // 
            2) / embed_dim)) for i in range(embed_dim)] for pos in range(
            max_seq_len)])
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])
        self.position_embed = nn.Parameter(torch.tensor(position_encodings),
            requires_grad=False)

    def forward(self, mask):
        """
        Args:
            mask: Use none zero as valid value flag and 0 as pad flag. Tensor[batch_size, max_seq_len]

        Return:
            Tensor[batch, max_seq_len, embed_dim]
        """
        mask = mask.unsqueeze(-1).expand(-1, -1, self.position_embed.shape[-1])
        embeddings = self.position_embed.unsqueeze(0).expand(mask.shape[0],
            -1, -1)
        return embeddings.masked_fill(mask == 0, 0).float()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'max_seq_len': 4}]
