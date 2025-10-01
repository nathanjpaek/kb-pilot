import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttentionLayer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, question, question_mask):
        qtn = question.view(-1, question.shape[-1])
        attn_scores = self.linear(qtn)
        attn_scores = attn_scores.view(question.shape[0], question.shape[1])
        attn_scores = attn_scores.masked_fill(question_mask == 1, -float('inf')
            )
        alpha = F.softmax(attn_scores, dim=1)
        return alpha


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
