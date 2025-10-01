import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn import Parameter


class ArchSampler(nn.Module):

    def __init__(self, distrib_dim, all_same, deter_eval, var_names=None, *
        args, **kwargs):
        super().__init__()
        self.distrib_dim = distrib_dim
        self.all_same = all_same
        self.deter_eval = deter_eval
        self.frozen = False
        self.log_probas = None
        self.distrib_entropies = None
        self._seq_probas = None
        if var_names is not None:
            assert len(var_names) == self.distrib_dim
        self.var_names = var_names

    def freeze(self):
        self.frozen = True

    def start_new_sequence(self):
        self.log_probas = []
        self.distrib_entropies = []
        self._seq_probas = []

    def nodes_to_prune(self, treshold):
        nodes = []
        for node, weight in zip(self.var_names, self().squeeze().unbind()):
            if weight < treshold:
                nodes.append(node)
        return nodes

    def sample_archs(self, batch_size, probas):
        """
        Hook called by pytorch before each forward
        :param: Current module
        :param input: Input given to the module's forward
        :return:
        """
        self._check_probas(probas, self.all_same)
        if probas.size(0) != batch_size:
            if probas.size(0) != 1:
                raise ValueError(
                    "Sampling probabilities dimensions {} doesn't match with batch size {}."
                    .format(probas.size(), batch_size))
            if not self.all_same:
                probas = probas.expand(batch_size, -1)
        distrib = torch.distributions.Bernoulli(probas)
        if not self.training and self.deter_eval:
            samplings = (probas > 0.5).float()
        else:
            samplings = distrib.sample()
        if self.all_same:
            samplings = samplings.expand(batch_size, -1)
        self._seq_probas.append(probas)
        self.distrib_entropies.append(distrib.entropy())
        self.log_probas.append(distrib.log_prob(samplings))
        return samplings

    def _check_probas(self, probas, all_same):
        """
        :param probas: B_size*N_nodes Tensor containing the probability of each
         arch being sampled in the nex forward.
        :param all_same: if True, the same sampling will be used for the whole
        batch in the next forward.
        :return:
        """
        if probas.dim() != 2 or all_same and probas.size(0) != 1:
            raise ValueError(
                'probas params has wrong dimension: {} (all_same={})'.
                format(probas.size(), all_same))
        if probas.size(-1) != self.distrib_dim:
            raise ValueError(
                'Should have exactly as many probas as the number of stochastic nodes({}), got {} instead.'
                .format(self.distrib_dim, probas.size(-1)))

    @property
    def last_arch_probas(self):
        return self.probas

    @property
    def last_sequence_probas(self):
        """
        :return: The probabilities of each arch for the last sequence in
        format (seq_len*batch_size*n_sampling_params)
        """
        return torch.stack(self._seq_probas)


class StaticArchGenerator(ArchSampler):

    def __init__(self, initial_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = Parameter(torch.Tensor(1, self.distrib_dim))
        logit = np.log(initial_p / (1 - initial_p)
            ) if initial_p < 1 else float('inf')
        weight_init.constant_(self.params, logit)

    def forward(self, z=None):
        if self.frozen and self.training:
            raise RuntimeError(
                'Trying to sample from a frozen distrib gen in training mode')
        return self.params.sigmoid()

    def entropy(self):
        distrib = torch.distributions.Bernoulli(self.params.sigmoid())
        return distrib.entropy()

    def remove_var(self, name):
        assert self.var_names
        self.distrib_dim -= 1
        remove_idx = self.var_names.index(name)
        self.var_names.remove(name)
        all_idx = torch.ones_like(self.params).bool()
        all_idx[0, remove_idx] = 0
        self.params = nn.Parameter(self.params[all_idx].unsqueeze(0))

    def is_deterministic(self):
        distrib = self()
        return torch.equal(distrib, distrib ** 2)


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'initial_p': 4, 'distrib_dim': 4, 'all_same': 4,
        'deter_eval': 4}]
