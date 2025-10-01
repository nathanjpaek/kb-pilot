import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class MultiHeadAttention(nn.Module):

    def __init__(self, in_dim, out_dim, out_heads, relation_dim=0, residual
        =False, projection=True, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_heads = out_heads
        self.relation_dim = relation_dim
        assert self.out_dim % self.out_heads == 0
        self.query_layer = nn.Linear(self.in_dim + self.relation_dim, self.
            out_dim, bias=False)
        self.key_layer = nn.Linear(self.in_dim + self.relation_dim, self.
            out_dim, bias=False)
        self.value_layer = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.residual = residual
        self.projection = projection
        if self.projection:
            self.proj_layer = nn.Linear(self.out_dim, self.out_dim)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln = nn.LayerNorm(self.out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.key_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)
        if self.projection:
            nn.init.uniform_(self.proj_layer.weight, -0.1, 0.1)

    def forward(self, query, key, relation=None, mask=None, key_mask=None,
        distance=None):
        """
        Args:
            query (torch.Tensor): [batch, query_len, in_dim]
            key (torch.Tensor): [batch, key_len, in_dim]
            relation (torch.Tensor): [batch, query_len, key_len, relation_dim]
            mask (torch.Tensor): [batch, query_len]
            key_mask (torch.Tensor): [batch, key_len]
        Returns:
            torch.Tensor: [batch, query_len, out_dim]
        """
        query_len = query.size(-2)
        key_len = key.size(-2)
        head_dim = self.out_dim // self.out_heads
        if key_mask is None:
            if torch.equal(query, key):
                key_mask = mask
        if relation is not None:
            relation = relation.view(-1, query_len, key_len, self.relation_dim)
            query_ = query.view(-1, query_len, 1, self.in_dim).repeat(1, 1,
                key_len, 1)
            query_ = torch.cat([query_, relation], dim=-1)
            key_ = key.view(-1, 1, key_len, self.in_dim).repeat(1,
                query_len, 1, 1)
            key_ = torch.cat([key_, relation], dim=-1)
            Q = self.query_layer(query_).view(-1, query_len * key_len, self
                .out_heads, head_dim)
            K = self.key_layer(key_).view(-1, query_len * key_len, self.
                out_heads, head_dim)
            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, key_len,
                head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, query_len, key_len,
                head_dim)
            attention = (Q * K).sum(dim=-1)
        else:
            Q = self.query_layer(query).view(-1, query_len, self.out_heads,
                head_dim)
            K = self.key_layer(key).view(-1, key_len, self.out_heads, head_dim)
            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, key_len, head_dim)
            attention = torch.bmm(Q, K.transpose(1, 2))
        if distance is not None:
            attention = attention - torch.log1p(distance.repeat(self.
                out_heads, 1, 1))
        attention = attention * float(head_dim) ** -0.5
        if key_mask is not None:
            attention = attention.view(-1, self.out_heads, query_len, key_len)
            attention = attention + ((1 - key_mask) * -1e+32).view(-1, 1, 1,
                key_len)
        attention = F.softmax(attention, dim=-1)
        if mask is not None:
            attention = attention * mask.view(-1, 1, query_len, 1)
            attention = attention.contiguous().view(-1, query_len, key_len)
        V = self.value_layer(key).view(-1, key_len, self.out_heads, head_dim)
        V = V.transpose(1, 2).contiguous().view(-1, key_len, head_dim)
        output = torch.bmm(attention, V).view(-1, self.out_heads, query_len,
            head_dim)
        output = output.transpose(1, 2).contiguous().view(*query.size()[:-2
            ], query_len, self.out_dim)
        if self.projection:
            output = self.proj_layer(output)
        if self.residual:
            output = output + query
        if self.layer_norm:
            output = self.ln(output)
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        attention = attention.view(*query.size()[:-2], self.out_heads,
            query_len, key_len).detach()
        return output, attention


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'out_heads': 4}]
