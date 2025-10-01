import torch
import torch.nn.functional as F


class N_TransE(torch.nn.Module):

    def __init__(self, p, params):
        super(N_TransE, self).__init__()
        self.p = p
        self.params = params

    def forward(self, e1, r, e2):
        pred = -torch.norm(e1 + r - e2, p=self.p, dim=1)
        return pred

    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score + self.params[0] - neg_score).sum(
            ) + self.params[1] * F.relu(pos_score - self.params[2]).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 4, 'params': 4}]
