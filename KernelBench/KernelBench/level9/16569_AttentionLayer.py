import torch
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


class AttentionLayer(nn.Module):

    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2 * output_embed_dim, output_embed_dim,
            bias=False)

    def forward(self, input, source_hids):
        x = self.input_proj(input)
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = F.softmax(attn_scores.t(), dim=1).t()
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = F.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_embed_dim': 4, 'output_embed_dim': 4}]
