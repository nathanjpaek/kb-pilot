import torch
import torch.nn as nn


class N_R_Align(torch.nn.Module):

    def __init__(self, params):
        super(N_R_Align, self).__init__()
        self.params = params
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-06)

    def forward(self, e1, e2, n1, n2):
        return self.params * torch.sigmoid(self.cos_sim(n1, n2)) + (1 -
            self.params) * torch.sigmoid(self.cos_sim(e1, e2))

    def loss(self, pos_score, neg_score, target):
        return -torch.log(pos_score).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'params': 4}]
