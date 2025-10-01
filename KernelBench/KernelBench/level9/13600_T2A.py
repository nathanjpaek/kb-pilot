import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.distributed


class T2A(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.U = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, article_hidden, template_hidden):
        seq_len = template_hidden.shape[0]
        article_hidden = article_hidden[-1, :, :].repeat(seq_len, 1, 1)
        s = self.W(article_hidden) + self.U(template_hidden) + self.b
        s = template_hidden * F.sigmoid(s)
        return s


def get_inputs():
    return [torch.rand([4, 16, 4, 4]), torch.rand([4, 64, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
