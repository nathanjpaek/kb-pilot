import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity


def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    """
    Return: [batch_size, decompse_dim, dim]
    """
    in_tensor = in_tensor.unsqueeze(1)
    decompose_params = decompose_params.unsqueeze(0)
    return torch.mul(in_tensor, decompose_params)


class AtteMatchLay(nn.Module):

    def __init__(self, mp_dim, cont_dim):
        super(AtteMatchLay, self).__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim
        self.register_parameter('weight', nn.Parameter(torch.Tensor(mp_dim,
            cont_dim)))
        self.weight.data.uniform_(-1.0, 1.0)

    def forward(self, repres, max_att):
        """
        Args:
            repres - [bsz, a_len|q_len, cont_dim]
            max_att - [bsz, q_len|a_len, cont_dim]
        Return:
            size - [bsz, sentence_len, mp_dim]
        """
        bsz = repres.size(0)
        sent_len = repres.size(1)
        repres = repres.view(-1, self.cont_dim)
        max_att = max_att.view(-1, self.cont_dim)
        repres = multi_perspective_expand_for_2D(repres, self.weight)
        max_att = multi_perspective_expand_for_2D(max_att, self.weight)
        temp = cosine_similarity(repres, max_att, repres.dim() - 1)
        return temp.view(bsz, sent_len, self.mp_dim)


def get_inputs():
    return [torch.rand([16, 4, 4]), torch.rand([64, 4])]


def get_init_inputs():
    return [[], {'mp_dim': 4, 'cont_dim': 4}]
