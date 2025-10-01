import torch
import torch.nn.functional as F
import torch.nn as nn


class AttnConnector(nn.Module):

    def __init__(self, rnn_cell, query_size, key_size, content_size,
        output_size, attn_size):
        super(AttnConnector, self).__init__()
        self.query_embed = nn.Linear(query_size, attn_size)
        self.key_embed = nn.Linear(key_size, attn_size)
        self.attn_w = nn.Linear(attn_size, 1)
        if rnn_cell == 'lstm':
            self.project_h = nn.Linear(content_size + query_size, output_size)
            self.project_c = nn.Linear(content_size + query_size, output_size)
        else:
            self.project = nn.Linear(content_size + query_size, output_size)
        self.rnn_cell = rnn_cell
        self.query_size = query_size
        self.key_size = key_size
        self.content_size = content_size
        self.output_size = output_size

    def forward(self, queries, keys, contents):
        batch_size = keys.size(0)
        num_key = keys.size(1)
        query_embeded = self.query_embed(queries)
        key_embeded = self.key_embed(keys)
        tiled_query = query_embeded.unsqueeze(1).repeat(1, num_key, 1)
        fc1 = F.tanh(tiled_query + key_embeded)
        attn = self.attn_w(fc1).squeeze(-1)
        attn = F.sigmoid(attn.view(-1, num_key)).view(batch_size, -1, num_key)
        mix = torch.bmm(attn, contents).squeeze(1)
        out = torch.cat([mix, queries], dim=1)
        if self.rnn_cell == 'lstm':
            h = self.project_h(out).unsqueeze(0)
            c = self.project_c(out).unsqueeze(0)
            new_s = h, c
        else:
            new_s = self.project(out).unsqueeze(0)
        return new_s


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'rnn_cell': 4, 'query_size': 4, 'key_size': 4,
        'content_size': 4, 'output_size': 4, 'attn_size': 4}]
