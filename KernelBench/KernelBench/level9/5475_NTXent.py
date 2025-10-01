import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXent(nn.Module):

    def __init__(self, metric: 'str'='CosineSimilarity', temperature:
        'float'=0.5, reduction: 'str'='mean'):
        super().__init__()
        if metric not in ['CosineSimilarity']:
            raise ValueError('Undefined metric!')
        if reduction not in ['AvgNonInf', 'mean', 'sum', 'none']:
            raise ValueError('Undefined reduction!')
        self.metric = metric
        self.t = temperature
        self.reduction = reduction

    def forward(self, embedding, label):
        if self.metric == 'CosineSimilarity':
            emb = F.normalize(embedding, dim=1)
            sim_mat = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0
                ), dim=2)
        sim_mat = torch.exp(sim_mat / self.t)
        pos = torch.sum(sim_mat * label, dim=1)
        inner = pos / torch.sum(sim_mat, dim=1)
        loss = -torch.log(inner)
        if self.reduction == 'AvgNonInf':
            non_inf = inner > 0
            loss = loss * non_inf
            loss = loss.sum() / torch.sum(non_inf) if torch.sum(non_inf
                ) > 0 else loss.mean()
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
