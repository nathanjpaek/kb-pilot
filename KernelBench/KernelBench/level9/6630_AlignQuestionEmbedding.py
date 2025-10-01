import torch
import torch.nn.functional as F
from torch import nn


class AlignQuestionEmbedding(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, context, question, question_mask):
        ctx_ = self.linear(context)
        ctx_ = self.relu(ctx_)
        qtn_ = self.linear(question)
        qtn_ = self.relu(qtn_)
        qtn_transpose = qtn_.permute(0, 2, 1)
        align_scores = torch.bmm(ctx_, qtn_transpose)
        qtn_mask = question_mask.unsqueeze(1).expand(align_scores.size())
        align_scores = align_scores.masked_fill(qtn_mask == 1, -float('inf'))
        align_scores_flat = align_scores.view(-1, question.size(1))
        alpha = F.softmax(align_scores_flat, dim=1)
        alpha = alpha.view(-1, context.shape[1], question.shape[1])
        align_embedding = torch.bmm(alpha, question)
        return align_embedding


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
