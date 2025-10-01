import torch
import torch.nn as nn


class GraphAttention(nn.Module):

    def __init__(self, d_q, d_v, alpha, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(d_q + d_v, 1)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, query, value, mask):
        """
        query - [batch_size, node_num * node_num, d_hidden]
        value - [batch_size, node_num * node_num, d_model]
        mask - [batch_size, node_num, node_num]
        """
        node_num = int(query.size(1) ** 0.5)
        query = query.view(-1, node_num, query.size(2))
        value = value.view(-1, node_num, value.size(2))
        pre_attention = torch.cat([query, value], dim=2)
        energy = self.leaky_relu(self.attention(pre_attention).squeeze(2))
        mask = mask.view(-1, node_num)
        zero_vec = -9000000000000000.0 * torch.ones_like(energy)
        try:
            attention = torch.where(mask > 0, energy, zero_vec)
        except:
            None
        scores = torch.softmax(attention, dim=1)
        scores = self.dropout(scores)
        value = torch.bmm(scores.unsqueeze(1), value).squeeze(1)
        value = value.view(-1, node_num, value.size(-1))
        return value


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_q': 4, 'd_v': 4, 'alpha': 4}]
