import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
import torch.nn.parallel
import torch.utils.data
import torch.distributions


def gumbel_softmax_sample(logits: 'torch.Tensor', temperature: 'float'=1.0,
    training: 'bool'=True, straight_through: 'bool'=False):
    size = logits.size()
    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).view(-1, size[-1])
        one_hot.scatter_(1, indexes.view(-1, 1), 1)
        one_hot = one_hot.view(*size)
        return one_hot
    sample = RelaxedOneHotCategorical(logits=logits, temperature=temperature
        ).rsample()
    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)
        sample = sample + (hard_sample - sample).detach()
    return sample


class GumbelSoftmaxLayer(nn.Module):

    def __init__(self, temperature: 'float'=1.0, trainable_temperature:
        'bool'=False, straight_through: 'bool'=False):
        super(GumbelSoftmaxLayer, self).__init__()
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(torch.tensor([temperature
                ]), requires_grad=True)

    def forward(self, logits: 'torch.Tensor'):
        return gumbel_softmax_sample(logits, self.temperature, self.
            training, self.straight_through)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
