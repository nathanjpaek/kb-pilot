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


class MaxpoolMatchLay(nn.Module):

    def __init__(self, mp_dim, cont_dim):
        super().__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim
        self.register_parameter('weight', nn.Parameter(torch.Tensor(mp_dim,
            cont_dim)))
        self.weight.data.uniform_(-1.0, 1.0)

    def forward(self, cont_repres, other_cont_repres):
        """
        Args:
            cont_repres - [batch_size, this_len, context_lstm_dim]
            other_cont_repres - [batch_size, other_len, context_lstm_dim]
        Return:
            size - [bsz, this_len, mp_dim*2]
        """
        bsz = cont_repres.size(0)
        this_len = cont_repres.size(1)
        other_len = other_cont_repres.size(1)
        cont_repres = cont_repres.view(-1, self.cont_dim)
        other_cont_repres = other_cont_repres.view(-1, self.cont_dim)
        cont_repres = multi_perspective_expand_for_2D(cont_repres, self.weight)
        other_cont_repres = multi_perspective_expand_for_2D(other_cont_repres,
            self.weight)
        cont_repres = cont_repres.view(bsz, this_len, self.mp_dim, self.
            cont_dim)
        other_cont_repres = other_cont_repres.view(bsz, other_len, self.
            mp_dim, self.cont_dim)
        cont_repres = cont_repres.unsqueeze(2)
        other_cont_repres = other_cont_repres.unsqueeze(1)
        simi = cosine_similarity(cont_repres, other_cont_repres, 
            cont_repres.dim() - 1)
        t_max, _ = simi.max(2)
        t_mean = simi.mean(2)
        return torch.cat((t_max, t_mean), 2)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'mp_dim': 4, 'cont_dim': 4}]
