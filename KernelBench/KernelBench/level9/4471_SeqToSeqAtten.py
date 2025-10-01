import torch
import torch.utils.data


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-06)
    return softmax


class SeqToSeqAtten(torch.nn.Module):
    """
    Args:
        -
    Inputs:
        - h1: (seq1_len, batch, hidden_size)
        - h1_mask: (batch, seq1_len)
        - h2: (seq2_len, batch, hidden_size)
        - h2_mask: (batch, seq2_len)
    Outputs:
        - output: (seq1_len, batch, hidden_size)
        - alpha: (batch, seq1_len, seq2_len)
    """

    def __init__(self):
        super(SeqToSeqAtten, self).__init__()

    def forward(self, h1, h2, h2_mask):
        h1 = h1.transpose(0, 1)
        h2 = h2.transpose(0, 1)
        alpha = h1.bmm(h2.transpose(1, 2))
        alpha = masked_softmax(alpha, h2_mask.unsqueeze(1), dim=2)
        alpha_seq2 = alpha.bmm(h2)
        alpha_seq2 = alpha_seq2.transpose(0, 1)
        return alpha_seq2, alpha


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
