import torch
import torch.nn as nn
import torch.multiprocessing
import torch.utils.data
import torch.nn.modules.loss


class Context2AnswerAttention(nn.Module):

    def __init__(self, dim, hidden_size):
        super(Context2AnswerAttention, self).__init__()
        self.linear_sim = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, context, answers, out_answers, mask_answers=None):
        """
        Parameters
        :context, (B, L, dim)
        :answers, (B, N, dim)
        :mask, (L, N)

        Returns
        :ques_emb, (L, dim)
        """
        context_fc = torch.relu(self.linear_sim(context))
        questions_fc = torch.relu(self.linear_sim(answers))
        attention = torch.matmul(context_fc, questions_fc.transpose(-1, -2))
        if mask_answers is not None:
            attention = attention.masked_fill(~mask_answers.unsqueeze(1).
                bool(), -Constants.INF)
        prob = torch.softmax(attention, dim=-1)
        emb = torch.matmul(prob, out_answers)
        return emb


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidden_size': 4}]
