import torch
import torch.nn as nn


class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout_rate=0.1,
        feedforward_size=256):
        """The core module with both spatial attention module and 
           temporal attention model embedded within it.
        """
        super(AttentionLayer, self).__init__()
        self.spatial_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.temporal_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, feedforward_size)
        self.linear2 = nn.Linear(feedforward_size, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        :param inputs: shape (T, B, H)
        :returns out: shape (T, B, H)
        """
        spatial_out, _spatial_attention_matrix = self.spatial_attention(inputs,
            inputs, inputs)
        spatial_out = self.dropout(spatial_out)
        spatial_out += inputs
        spatial_out = self.layer_norm(spatial_out)
        temporal_out, _temporal_attention_matrix = self.temporal_attention(
            inputs, inputs, inputs)
        temporal_out = self.dropout(temporal_out)
        temporal_out += inputs
        temporal_out = self.layer_norm(temporal_out)
        attention_out = spatial_out + temporal_out
        out = self.linear1(attention_out)
        out = self.linear2(out)
        out = self.dropout(out)
        out += attention_out
        out = self.layer_norm(out)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'num_heads': 4}]
