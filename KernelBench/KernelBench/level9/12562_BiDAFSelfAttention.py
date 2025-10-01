import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e+30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)
    return probs


class BiDAFSelfAttention(nn.Module):

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFSelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.c_weight_2 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight_2 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight_2 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.c_weight_3 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight_3 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight_3 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.c_weight_4 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight_4 = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight_4 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight, self.
            c_weight_2, self.q_weight_2, self.cq_weight_2, self.c_weight_3,
            self.q_weight_3, self.cq_weight_3, self.c_weight_4, self.
            q_weight_4, self.cq_weight_4):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, c_mask):
        batch_size, c_len, _ = c.size()
        s = self.get_similarity_matrix(c, c, self.c_weight, self.q_weight,
            self.cq_weight)
        c_mask = c_mask.view(batch_size, c_len, 1)
        s_softmax = masked_softmax(s, c_mask, dim=2)
        a = torch.bmm(s_softmax, c)
        s_2 = self.get_similarity_matrix(c, c, self.c_weight_2, self.
            q_weight_2, self.cq_weight_2)
        s_softmax_2 = masked_softmax(s_2, c_mask, dim=2)
        a_2 = torch.bmm(s_softmax_2, c)
        s_3 = self.get_similarity_matrix(c, c, self.c_weight_3, self.
            q_weight_3, self.cq_weight_3)
        s_softmax_3 = masked_softmax(s_3, c_mask, dim=2)
        a_3 = torch.bmm(s_softmax_3, c)
        s_4 = self.get_similarity_matrix(c, c, self.c_weight_4, self.
            q_weight_4, self.cq_weight_4)
        s_softmax_4 = masked_softmax(s_4, c_mask, dim=2)
        a_4 = torch.bmm(s_softmax_4, c)
        x = torch.cat([c, a, a_2, a_3, a_4], dim=2)
        return x

    def get_similarity_matrix(self, c, q, c_weight, q_weight, cq_weight):
        """ Just performing w_sim^T[c_i; q_j; c_i * q_j] except c == q
        (Copied over from BidafAttention)
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)
        q = F.dropout(q, self.drop_prob, self.training)
        s0 = torch.matmul(c, c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, q_weight).transpose(1, 2).expand([-1, c_len, -1])
        s2 = torch.matmul(c * cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias
        return s


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 1])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
