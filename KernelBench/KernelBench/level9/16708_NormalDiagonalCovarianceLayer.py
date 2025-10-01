import abc
import math
import torch


class ProbabilisticLayer(torch.nn.Module, metaclass=abc.ABCMeta):
    """Probabilistic layer to be used by the encoder/decoder of a
    Variational AutoEncoder.

    """

    @abc.abstractmethod
    def forward(self, inputs):
        """Compute the parameters of the distribution conditioned on the
        input.

        Args:
            inputs (``torch.Tensor[N,dim]``): Conditional inputs.

        Returns:
            params (object): Parameters of the distribution.

        """
        pass

    @abc.abstractmethod
    def samples_and_llh(self, params, use_mean=False):
        """Samples using the reparameterization trick so that the each
        sample can be backpropagated trough. In addition, returns the
        log-likelihood of the samples given the sampling distribution.

        Args:
            params (object): Parameters of the sampling distribution.
            use_mean (boolean): If true, by pass the sampling and
                just return the mean value of the distribution.

        Returns:
            samples (``torch.Tensor``): Sampled values.
            llh (``torch.Tensor``): Log-likelihood for each sample.
        """
        pass

    @abc.abstractmethod
    def log_likelihood(self, data, params):
        """Log-likelihood of the data.

        Args:
            data (``torch.Tensor[N,dim]``): Data.
            params (object): Parameters of the distribution.

        """
        pass


class NormalDiagonalCovarianceLayer(ProbabilisticLayer):
    """Normal distribution with diagonal covariance matrix layer."""

    def __init__(self, dim_in, dim_out, variance_nonlinearity=None):
        super().__init__()
        self.mean = torch.nn.Linear(dim_in, dim_out)
        self.logvar = torch.nn.Linear(dim_in, dim_out)
        if variance_nonlinearity is None:
            variance_nonlinearity = torch.nn.Softplus()
        self.variance_nonlinearity = variance_nonlinearity

    def forward(self, inputs):
        return self.mean(inputs), self.variance_nonlinearity(self.logvar(
            inputs))

    def samples_and_llh(self, params, use_mean=False):
        means, variances = params
        if use_mean:
            samples = means
        else:
            dtype, device = means.dtype, means.device
            noise = torch.randn(*means.shape, dtype=dtype, device=device)
            std_dev = variances.sqrt()
            samples = means + std_dev * noise
        llhs = self.log_likelihood(samples, params)
        return samples, llhs

    def log_likelihood(self, data, params):
        means, variances = params
        dim = means.shape[-1]
        delta = torch.sum((data - means).pow(2) / variances, dim=-1)
        return -0.5 * (variances.log().sum(dim=-1) + delta + dim * math.log
            (2 * math.pi))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
