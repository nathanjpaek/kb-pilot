import torch
import torch.nn as nn


class CosAttention(nn.Module):

    def __init__(self):
        super(CosAttention, self).__init__()

    def forward(self, q, k, v):
        """
        q: (batchsize, hidden_dim)
        k: (batchsize, seqlen, hidden_dim)
        v: (batchsize, seqlen, hidden_dim)
        """
        seq_len = k.size()[1]
        q_output = q.unsqueeze(1).repeat(1, seq_len, 1)
        cos_sim = torch.cosine_similarity(q_output, k, -1)
        cos_sim = cos_sim.unsqueeze(-1)
        outputs = v * cos_sim
        return outputs


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4,
        4, 4])]


def get_init_inputs():
    return [[], {}]
