import torch
import torch.nn as nn
import torch.nn.functional as F


class DirichletLayer(nn.Module):

    def __init__(self, evidence='exp', dim=-1):
        super(DirichletLayer, self).__init__()
        self.evidence = evidence
        self.dim = dim

    def evidence_func(self, logit):
        if self.evidence == 'relu':
            return F.relu(logit)
        if self.evidence == 'exp':
            return torch.exp(torch.clamp(logit, -10, 10))
        if self.evidence == 'softplus':
            return F.softplus(logit)

    def compute_uncertainty(self, logit):
        num_classes = logit.size(-1)
        alpha = self.evidence_func(logit) + 1
        uncertainty = num_classes / alpha.sum(-1)
        return uncertainty

    def forward(self, logit):
        alpha = self.evidence_func(logit) + 1
        conf = alpha / alpha.sum(dim=self.dim, keepdim=True)
        return conf


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
