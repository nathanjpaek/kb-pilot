from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical


class MixtureDensityHead(nn.Module):

    def __init__(self, config: 'DictConfig', **kwargs):
        self.hparams = config
        super().__init__()
        self._build_network()

    def _build_network(self):
        self.pi = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.pi.weight)
        self.sigma = nn.Linear(self.hparams.input_dim, self.hparams.
            num_gaussian, bias=self.hparams.sigma_bias_flag)
        self.mu = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.mu.weight)
        if self.hparams.mu_bias_init is not None:
            for i, bias in enumerate(self.hparams.mu_bias_init):
                nn.init.constant_(self.mu.bias[i], bias)

    def forward(self, x):
        pi = self.pi(x)
        sigma = self.sigma(x)
        sigma = nn.ELU()(sigma) + 1 + 1e-15
        mu = self.mu(x)
        return pi, sigma, mu

    def gaussian_probability(self, sigma, mu, target, log=False):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.expand_as(sigma)
        if log:
            ret = -torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((
                target - mu) / sigma, 2)
        else:
            ret = ONEOVERSQRT2PI / sigma * torch.exp(-0.5 * ((target - mu) /
                sigma) ** 2)
        return ret

    def log_prob(self, pi, sigma, mu, y):
        log_component_prob = self.gaussian_probability(sigma, mu, y, log=True)
        log_mix_prob = torch.log(nn.functional.gumbel_softmax(pi, tau=self.
            hparams.softmax_temperature, dim=-1) + 1e-15)
        return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)

    def sample(self, pi, sigma, mu):
        """Draw samples from a MoG."""
        categorical = Categorical(pi)
        pis = categorical.sample().unsqueeze(1)
        sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
        sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
        return sample

    def generate_samples(self, pi, sigma, mu, n_samples=None):
        if n_samples is None:
            n_samples = self.hparams.n_samples
        samples = []
        softmax_pi = nn.functional.gumbel_softmax(pi, tau=self.hparams.
            softmax_temperature, dim=-1)
        assert (softmax_pi < 0).sum().item(
            ) == 0, 'pi parameter should not have negative'
        for _ in range(n_samples):
            samples.append(self.sample(softmax_pi, sigma, mu))
        samples = torch.cat(samples, dim=1)
        return samples

    def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
        samples = self.generate_samples(pi, sigma, mu, n_samples)
        if self.hparams.central_tendency == 'mean':
            y_hat = torch.mean(samples, dim=-1)
        elif self.hparams.central_tendency == 'median':
            y_hat = torch.median(samples, dim=-1).values
        return y_hat.unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(input_dim=4, num_gaussian=4,
        sigma_bias_flag=4, mu_bias_init=[4, 4])}]
