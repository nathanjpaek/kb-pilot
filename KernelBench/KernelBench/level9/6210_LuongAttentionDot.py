import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttentionDot(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, query, values):
        query = torch.squeeze(query, 0)
        query = torch.unsqueeze(query, 1)
        query_transposed = query.transpose(2, 1)
        score = torch.matmul(values, query_transposed)
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
