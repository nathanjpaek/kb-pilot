import torch
from torch import Tensor


class NeuralSort(torch.nn.Module):

    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, input: 'Tensor', scores: 'Tensor', cuda=None):
        """

        :param input:
        :param scores: logits of the scores by which the elements in input should be sorted.
        :param cuda:
        :return:
        """
        cuda = input.is_cuda if cuda is None else cuda
        dv = 'cuda' if cuda else 'cpu'
        scores = scores.unsqueeze(-1)
        bsize, dim = scores.size()[:2]
        one = torch.ones(dim, 1, device=dv)
        scores = torch.exp(scores)
        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0,
            1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim, device=dv) + 1)).type(torch
            .float)
        C = torch.matmul(scores, scaling.unsqueeze(0))
        P_max = (C - B).permute(0, 2, 1)
        P_hat_raw = P_max / self.tau
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)
        if self.hard:
            P = torch.zeros_like(P_hat, device=dv)
            b_idx = torch.arange(bsize, device=dv).repeat([1, dim]).view(dim,
                bsize)
            b_idx = b_idx.transpose(dim0=1, dim1=0).flatten().type(torch.long)
            r_idx = torch.arange(dim, device=dv).repeat([bsize, 1]).flatten(
                ).type(torch.long)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()
            brc_idx = torch.stack((b_idx, r_idx, c_idx))
            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        _b, _s, _z = input.size()
        out = torch.bmm(P_hat, input)
        return out, P_hat, P_hat_raw


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
