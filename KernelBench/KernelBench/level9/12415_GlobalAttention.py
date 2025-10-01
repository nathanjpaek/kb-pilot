import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    """
    Global Attention between encoder and decoder
    """

    def __init__(self, key_features, query_features, value_features,
        hidden_features=None, dropout=0.0):
        """

        Args:
            key_features: int
                dimension of keys
            query_features: int
                dimension of queries
            value_features: int
                dimension of values (outputs)
            hidden_features: int
                dimension of hidden states (default value_features)
            dropout: float
                dropout rate
        """
        super(GlobalAttention, self).__init__()
        if hidden_features is None:
            hidden_features = value_features
        self.key_proj = nn.Linear(key_features, 2 * hidden_features, bias=True)
        self.query_proj = nn.Linear(query_features, hidden_features, bias=True)
        self.dropout = dropout
        self.fc = nn.Linear(hidden_features, value_features)
        self.hidden_features = hidden_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, query, key, key_mask=None):
        """

        Args:
            query: Tensor
                query tensor [batch, query_length, query_features]
            key: Tensor
                key tensor [batch, key_length, key_features]
            key_mask: ByteTensor or None
                binary ByteTensor [batch, src_len] padding elements are indicated by 1s.

        Returns: Tensor
            value tensor [batch, query_length, value_features]

        """
        bs, timesteps, _ = key.size()
        dim = self.hidden_features
        query = self.query_proj(query)
        c = self.key_proj(key)
        c = c.view(bs, timesteps, 2, dim)
        key = c[:, :, 0]
        value = c[:, :, 1]
        attn_weights = torch.bmm(query, key.transpose(1, 2))
        if key_mask is not None:
            attn_weights = attn_weights.masked_fill(key_mask.unsqueeze(1),
                float('-inf'))
        attn_weights = F.softmax(attn_weights.float(), dim=-1, dtype=torch.
            float32 if attn_weights.dtype == torch.float16 else
            attn_weights.dtype)
        out = torch.bmm(attn_weights, value)
        out = F.dropout(self.fc(out), p=self.dropout, training=self.training)
        return out

    def init(self, query, key, key_mask=None, init_scale=1.0):
        with torch.no_grad():
            return self(query, key, key_mask=key_mask)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'key_features': 4, 'query_features': 4, 'value_features': 4}]
