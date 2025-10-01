import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.onnx.operators


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class LSTMAttentionLayer(nn.Module):

    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim,
        bias=False, dropout=0.0):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim,
            output_embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, input, source_hids, encoder_padding_mask=None,
        enc_dec_attn_constraint_mask=None):
        x = self.input_proj(input)
        attn_weights = torch.bmm(x.transpose(0, 1), source_hids.transpose(0,
            1).transpose(1, 2))
        if encoder_padding_mask is not None:
            attn_weights = attn_weights.float().masked_fill_(
                encoder_padding_mask.unsqueeze(1), float('-inf')).type_as(
                attn_weights)
        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.float().masked_fill_(
                enc_dec_attn_constraint_mask.bool(), float('-inf')).type_as(
                attn_weights)
        attn_logits = attn_weights
        sz = attn_weights.size()
        attn_scores = F.softmax(attn_weights.view(sz[0] * sz[1], sz[2]), dim=1)
        attn_scores = attn_scores.view(sz)
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.
            training)
        attn = torch.bmm(attn_scores, source_hids.transpose(0, 1)).transpose(
            0, 1)
        x = torch.tanh(self.output_proj(torch.cat((attn, input), dim=-1)))
        return x, attn_scores, attn_logits


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_embed_dim': 4, 'source_embed_dim': 4,
        'output_embed_dim': 4}]
