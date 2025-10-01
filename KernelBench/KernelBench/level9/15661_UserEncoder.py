from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(torch.nn.Module):
    """
    A general additive attention module.
    Originally for NAML.
    """

    def __init__(self, query_vector_dim, candidate_vector_dim, writer=None,
        tag=None, names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(torch.empty(
            query_vector_dim).uniform_(-0.1, 0.1))
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(temp, self.
            attention_query_vector), dim=1)
        if self.writer is not None:
            assert candidate_weights.size(1) == len(self.names)
            if self.local_step % 10 == 0:
                self.writer.add_scalars(self.tag, {x: y for x, y in zip(
                    self.names, candidate_weights.mean(dim=0))}, self.
                    local_step)
            self.local_step += 1
        target = torch.bmm(candidate_weights.unsqueeze(dim=1), candidate_vector
            ).squeeze(dim=1)
        return target


class UserEncoder(torch.nn.Module):

    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
            config.num_filters)

    def forward(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        user_vector = self.additive_attention(clicked_news_vector)
        return user_vector


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(query_vector_dim=4, num_filters=4)}]
