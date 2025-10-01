import torch
import torch.nn as nn


class ROUGH_FILTER(nn.Module):

    def __init__(self, user_num, embedding_size):
        super(ROUGH_FILTER, self).__init__()
        self.in_user_embedding = nn.Embedding(user_num, embedding_size)

    def forward(self, out_user_embedding_weight):
        score = torch.mm(self.in_user_embedding.weight,
            out_user_embedding_weight.permute(1, 0))
        score = torch.tanh(score)
        score = torch.relu(score)
        return score


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'user_num': 4, 'embedding_size': 4}]
