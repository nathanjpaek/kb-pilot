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


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_size': 4}]
