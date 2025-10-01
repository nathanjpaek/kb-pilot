import torch
import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter


class ScModel(nn.Module):
    """ Model for single cell data """

    def __init__(self, n_genes: 'int', n_celltypes: 'int', device: 't.device'
        ) ->None:
        super().__init__()
        self.K = n_celltypes
        self.G = n_genes
        self.theta = Parameter(t.Tensor(self.G, self.K))
        self.R = t.Tensor(self.G, self.K)
        self.o = Parameter(t.Tensor(self.G, 1))
        nn.init.normal_(self.o, mean=0.0, std=1.0)
        nn.init.normal_(self.theta, mean=0.0, std=1.0)
        self.nb = t.distributions.NegativeBinomial
        self.softpl = nn.functional.softplus
        self.logsig = nn.functional.logsigmoid

    def _llnb(self, x: 't.Tensor', meta: 't.LongTensor', sf: 't.Tensor'
        ) ->t.Tensor:
        """Log Likelihood for NB-model

        Returns the log likelihood for rates and logodds
        taken as a function of the observed counts.
        Assumes that single cell data is negative
        binomial distributed.

        Returns
        -------
        The log likelihood

        """
        log_unnormalized_prob = sf * self.R[:, meta] * self.logsig(-self.o
            ) + x * self.logsig(self.o)
        log_normalization = -t.lgamma(sf * self.R[:, meta] + x) + t.lgamma(
            1.0 + x) + t.lgamma(sf * self.R[:, meta])
        ll = t.sum(log_unnormalized_prob - log_normalization)
        return ll

    def forward(self, x: 't.Tensor', meta: 't.LongTensor', sf: 't.Tensor',
        **kwargs) ->t.Tensor:
        """Forward pass during optimization"""
        self.R = self.softpl(self.theta)
        self.loss = -self._llnb(x.transpose(1, 0), meta, sf)
        return self.loss

    def __str__(self):
        return 'sc_model'


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=
        torch.int64), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_genes': 4, 'n_celltypes': 4, 'device': 0}]
