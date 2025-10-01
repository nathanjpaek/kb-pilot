import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, dims, method='general', dropout=0.0):
        super().__init__()
        if method not in ('dot', 'general'):
            raise ValueError('Invalid attention type selected')
        self.method = method
        if method == 'general':
            self.linear_in = nn.Linear(dims, dims, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(dims * 2, dims, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        batch_size, output_len, dims = query.size()
        query_len = context.size(1)
        if self.method == 'general':
            query = query.contiguous()
            query = query.view(batch_size * output_len, dims)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dims)
        attention_scores = torch.bmm(query, context.transpose(1, 2).
            contiguous())
        attention_scores = attention_scores.view(batch_size * output_len,
            query_len)
        attention_weights = self.softmax(attention_scores)
        if self.dropout.p != 0.0:
            attention_weights = self.dropout(attention_weights)
        attention_weights = attention_weights.view(batch_size, output_len,
            query_len)
        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dims)
        output = self.linear_out(combined).view(batch_size, output_len, dims)
        output = self.tanh(output)
        return output, attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dims': 4}]
