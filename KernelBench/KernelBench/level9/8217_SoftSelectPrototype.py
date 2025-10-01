import torch
import torch.nn as nn


class SoftSelectAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SoftSelectAttention, self).__init__()

    def forward(self, support, query):
        """
        :param support: [few, dim]
        :param query: [batch, dim]
        :return:
        """
        query_ = query.unsqueeze(1).expand(query.size(0), support.size(0),
            query.size(1)).contiguous()
        support_ = support.unsqueeze(0).expand_as(query_).contiguous()
        scalar = support.size(1) ** -0.5
        score = torch.sum(query_ * support_, dim=2) * scalar
        att = torch.softmax(score, dim=1)
        center = torch.mm(att, support)
        return center


class SoftSelectPrototype(nn.Module):

    def __init__(self, r_dim):
        super(SoftSelectPrototype, self).__init__()
        self.Attention = SoftSelectAttention(hidden_size=r_dim)

    def forward(self, support, query):
        center = self.Attention(support, query)
        return center


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'r_dim': 4}]
