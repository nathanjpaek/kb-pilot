import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(self, annot_dim, query_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, annots, query):
        """
        Shapes:
            - annots: (batch, max_time, dim)
            - query: (batch, 1, dim) or (batch, dim)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)
        alignment = self.v(torch.tanh(processed_query + processed_annots))
        return alignment.squeeze(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'annot_dim': 4, 'query_dim': 4, 'attn_dim': 4}]
