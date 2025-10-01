import torch
import torch.nn.functional as F
import torch.nn as nn


class Attention(nn.Module):
    """
    Computing the attention over the words
    """

    def __init__(self, input_dim, proj_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.head = nn.Parameter(torch.Tensor(proj_dim, 1).uniform_(-0.1, 0.1))
        self.proj = nn.Linear(input_dim, proj_dim)

    def forward(self, input, input_mask):
        """
        input: batch, max_text_len, input_dim
        input_mask: batch, max_text_len
        """
        batch, max_input_len, _input_dim = input.size()
        proj_input = torch.tanh(self.proj(input.view(batch * max_input_len,
            -1)))
        att = torch.mm(proj_input, self.head)
        att = att.view(batch, max_input_len, 1)
        log_att = F.log_softmax(att, dim=1)
        att = F.softmax(att, dim=1)
        output = input * att * input_mask.unsqueeze(-1).detach()
        output = output.sum(dim=1)
        return output, att.squeeze(2), log_att.squeeze(2)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'proj_dim': 4}]
