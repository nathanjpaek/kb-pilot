import torch
import torch.nn.functional as F


class AlignEA(torch.nn.Module):

    def __init__(self, p, feat_drop, params):
        super(AlignEA, self).__init__()
        self.params = params

    def forward(self, e1, r, e2):
        return torch.sum(torch.pow(e1 + r - e2, 2), 1)

    def only_pos_loss(self, e1, r, e2):
        return -F.logsigmoid(-torch.sum(torch.pow(e1 + r - e2, 2), 1)).sum()

    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score - self.params[0]).sum() + self.params[1
            ] * F.relu(self.params[2] - neg_score).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 4, 'feat_drop': 4, 'params': 4}]
