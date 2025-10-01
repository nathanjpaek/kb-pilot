import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim
import torch.utils.data


class SparseGate(nn.Module):

    def __init__(self, in_features, n_experts, k=2):
        """
        Returns a sparsely gated noisy softmax.
        See OUTRAGEOUSLY LARGE NEURAL NETWORKS:
            THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER
            Shazeer et. al
            Link: https://arxiv.org/pdf/1701.06538.pdf
        """
        assert k > 1, 'Need k >= 1. If k == 1, then derivatives are zero everywhere.'
        super(SparseGate, self).__init__()
        self.gate_weights = Parameter(torch.Tensor(n_experts, in_features))
        self.noise_weights = Parameter(torch.Tensor(n_experts, in_features))
        self.n_experts = n_experts
        self.n_selected = k
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.size(0)
        noise = x.new_empty((batch_size, self.n_experts)).normal_()
        expert_weights = F.linear(x, self.gate_weights, None
            ) + noise * F.softplus(F.linear(x, self.noise_weights, None))
        top_k, indices = torch.topk(expert_weights, self.n_selected)
        top_k_softmax = F.softmax(top_k, dim=1)
        res = x.new_full((batch_size, self.n_experts), 0.0)
        return res.scatter_(1, indices, top_k_softmax)

    def reset_parameters(self):
        nn.init.constant_(self.gate_weights, 0.0)
        nn.init.constant_(self.noise_weights, 0.0)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'n_experts': 4}]
