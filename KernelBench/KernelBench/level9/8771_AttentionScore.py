import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionScore(nn.Module):
    """
    correlation_func = 1, sij = x1^Tx2
    correlation_func = 2, sij = (Wx1)D(Wx2)
    correlation_func = 3, sij = Relu(Wx1)DRelu(Wx2)
    correlation_func = 4, sij = x1^TWx2
    correlation_func = 5, sij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, input_size, hidden_size, correlation_func=1,
        do_similarity=False):
        super(AttentionScore, self).__init__()
        self.correlation_func = correlation_func
        self.hidden_size = hidden_size
        if correlation_func == 2 or correlation_func == 3:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)
            if do_similarity:
                self.diagonal = Parameter(torch.ones(1, 1, 1) / hidden_size **
                    0.5, requires_grad=False)
            else:
                self.diagonal = Parameter(torch.ones(1, 1, hidden_size),
                    requires_grad=True)
        if correlation_func == 4:
            self.linear = nn.Linear(input_size, input_size, bias=False)
        if correlation_func == 5:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x1, x2, x2_mask):
        """
        Input:
        x1: batch x word_num1 x dim
        x2: batch x word_num2 x dim
        Output:
        scores: batch x word_num1 x word_num2
        """
        x1_rep = x1
        x2_rep = x2
        batch = x1_rep.size(0)
        word_num1 = x1_rep.size(1)
        word_num2 = x2_rep.size(1)
        dim = x1_rep.size(2)
        if self.correlation_func == 2 or self.correlation_func == 3:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch,
                word_num1, self.hidden_size)
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch,
                word_num2, self.hidden_size)
            if self.correlation_func == 3:
                x1_rep = F.relu(x1_rep)
                x2_rep = F.relu(x2_rep)
            x1_rep = x1_rep * self.diagonal.expand_as(x1_rep)
        if self.correlation_func == 4:
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch,
                word_num2, dim)
        if self.correlation_func == 5:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch,
                word_num1, self.hidden_size)
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch,
                word_num2, self.hidden_size)
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)
        scores = x1_rep.bmm(x2_rep.transpose(1, 2))
        empty_mask = x2_mask.eq(0).expand_as(scores)
        scores.data.masked_fill_(empty_mask.data, -float('inf'))
        alpha_flat = F.softmax(scores, dim=-1)
        return alpha_flat


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
