import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import BatchSampler
from torch.utils.data import SubsetRandomSampler


class RecurrentNeuralRegressor(nn.Module):

    def __init__(self, sizes, prior, nonlin='relu'):
        super(RecurrentNeuralRegressor, self).__init__()
        self.sizes = sizes
        self.nb_states = self.sizes[-1]
        self.prior = prior
        nlist = dict(relu=F.relu, tanh=F.tanh, softmax=F.log_softmax,
            linear=F.linear)
        self.nonlin = nlist[nonlin]
        self.layer = nn.Linear(self.sizes[0], self.sizes[1])
        self.output = nn.Linear(self.sizes[1], self.sizes[2])
        _mat = 0.95 * torch.eye(self.nb_states) + 0.05 * torch.rand(self.
            nb_states, self.nb_states)
        _mat /= torch.sum(_mat, dim=1, keepdim=True)
        self.logmat = nn.Parameter(torch.log(_mat))
        self.optim = None

    def log_prior(self):
        lp = 0.0
        if self.prior:
            _matrix = torch.exp(self.logmat - torch.logsumexp(self.logmat,
                dim=-1, keepdim=True))
            for k in range(self.nb_states):
                alpha = self.prior['alpha'] * torch.ones(self.nb_states
                    ) + self.prior['kappa'] * torch.as_tensor(torch.arange(
                    self.nb_states) == k, dtype=torch.float32)
                _dirichlet = torch.distributions.dirichlet.Dirichlet(alpha)
                lp += _dirichlet.log_prob(_matrix[k])
        return lp

    def forward(self, xu):
        out = self.output(self.nonlin(self.layer(xu)))
        _logtrans = self.logmat[None, :, :] + out[:, None, :]
        return _logtrans - torch.logsumexp(_logtrans, dim=-1, keepdim=True)

    def elbo(self, zeta, xu):
        logtrans = self.forward(xu)
        return torch.sum(zeta * logtrans) + self.log_prior()

    def fit(self, zeta, xu, nb_iter=100, batch_size=None, lr=0.001):
        self.optim = Adam(self.parameters(), lr=lr)
        batch_size = xu.shape[0] if batch_size is None else batch_size
        batches = list(BatchSampler(SubsetRandomSampler(range(xu.shape[0])),
            batch_size, False))
        for n in range(nb_iter):
            for batch in batches:
                self.optim.zero_grad()
                loss = -self.elbo(zeta[batch], xu[batch])
                loss.backward()
                self.optim.step()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sizes': [4, 4, 4], 'prior': 4}]
