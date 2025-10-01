import torch
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing
import torch.nn.modules.loss
from scipy.sparse import *


class Context2AnswerAttention(nn.Module):

    def __init__(self, dim, hidden_size):
        super(Context2AnswerAttention, self).__init__()
        self.linear_sim = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, context, answers, out_answers, ans_mask=None):
        """
        Parameters
        :context, (batch_size, L, dim)
        :answers, (batch_size, N, dim)
        :out_answers, (batch_size, N, dim)
        :ans_mask, (batch_size, N)

        Returns
        :ques_emb, (batch_size, L, dim)
        """
        context_fc = torch.relu(self.linear_sim(context))
        questions_fc = torch.relu(self.linear_sim(answers))
        attention = torch.matmul(context_fc, questions_fc.transpose(-1, -2))
        if ans_mask is not None:
            attention = attention.masked_fill_((1 - ans_mask).bool().
                unsqueeze(1), -INF)
        prob = torch.softmax(attention, dim=-1)
        ques_emb = torch.matmul(prob, out_answers)
        return ques_emb


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidden_size': 4}]
