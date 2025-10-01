import abc
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


class BernoulliLayer(ProbabilisticLayer):
    """Bernoulli distribution layer."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mean = torch.nn.Linear(dim_in, dim_out)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        return self.sigmoid(self.mean(inputs))

    def samples_and_llh(self, params, use_mean=False):
        """The Bernoulli layer cannot be used as an encoding
        distribution since we cannot backpropagate through discrete
        samples.

        """
        raise NotImplementedError

    def log_likelihood(self, data, params):
        means = params
        epsilon = 1e-06
        per_pixel_bce = data * torch.log(epsilon + means) + (1.0 - data
            ) * torch.log(epsilon + 1 - means)
        return per_pixel_bce.sum(dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
