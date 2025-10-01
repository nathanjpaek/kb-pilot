import math
import torch
import torch.nn as nn
import torch.utils.data


class GCNLayer(nn.Module):

    def __init__(self, embed_size, dropout=0.0):
        super().__init__()
        self.embed_size = embed_size
        self.ctx_layer = nn.Linear(self.embed_size, self.embed_size, bias=False
            )
        self.layernorm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_fts, rel_edges):
        """Args:
      node_fts: (batch_size, num_nodes, embed_size)
      rel_edges: (batch_size, num_nodes, num_nodes)
    """
        ctx_embeds = self.ctx_layer(torch.bmm(rel_edges, node_fts))
        node_embeds = node_fts + self.dropout(ctx_embeds)
        node_embeds = self.layernorm(node_embeds)
        return node_embeds


class AttnGCNLayer(GCNLayer):

    def __init__(self, embed_size, d_ff, dropout=0.0):
        super().__init__(embed_size, dropout=dropout)
        self.edge_attn_query = nn.Linear(embed_size, d_ff)
        self.edge_attn_key = nn.Linear(embed_size, d_ff)
        self.attn_denominator = math.sqrt(d_ff)

    def forward(self, node_fts, rel_edges):
        """
    Args:
      node_fts: (batch_size, num_nodes, embed_size)
      rel_edges: (batch_size, num_nodes, num_nodes)
    """
        attn_scores = torch.einsum('bod,bid->boi', self.edge_attn_query(
            node_fts), self.edge_attn_key(node_fts)) / self.attn_denominator
        attn_scores = attn_scores.masked_fill(rel_edges == 0, -1e+18)
        attn_scores = torch.softmax(attn_scores, dim=2)
        attn_scores = attn_scores.masked_fill(rel_edges == 0, 0)
        ctx_embeds = self.ctx_layer(torch.bmm(attn_scores, node_fts))
        node_embeds = node_fts + self.dropout(ctx_embeds)
        node_embeds = self.layernorm(node_embeds)
        return node_embeds


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_size': 4, 'd_ff': 4}]
