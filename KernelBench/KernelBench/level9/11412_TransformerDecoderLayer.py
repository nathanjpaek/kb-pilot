import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.distributions


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block. Follows an implementation in fairseq with args.decoder_normalize_before=True,
    i.e. order of operations is different from those in the original paper.
    """

    def __init__(self, num_heads, embed_dim, hidden_size, dropout=0.0,
        attention_dropout=0.0, activation_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.
            embed_dim, num_heads=num_heads, dropout=attention_dropout)
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.encoder_attn = torch.nn.MultiheadAttention(embed_dim=self.
            embed_dim, num_heads=num_heads, dropout=attention_dropout)
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.fc1 = torch.nn.Linear(self.embed_dim, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, self.embed_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x, encoder_out, key_mask=None, attn_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=
            key_mask, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn = self.encoder_attn(query=x, key=encoder_out, value=encoder_out
            )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x, attn


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_heads': 4, 'embed_dim': 4, 'hidden_size': 4}]
