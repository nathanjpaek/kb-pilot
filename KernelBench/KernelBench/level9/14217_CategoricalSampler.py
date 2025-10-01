import torch
import torch.nn as nn


class Sampler(nn.Module):
    """ args; logits: (batch, n_nodes)
		return; next_node: (batch, 1)
		TopKSampler <=> greedy; sample one with biggest probability
		CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
	"""

    def __init__(self, n_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples


class CategoricalSampler(Sampler):

    def forward(self, logits):
        return torch.multinomial(logits.exp(), self.n_samples)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
