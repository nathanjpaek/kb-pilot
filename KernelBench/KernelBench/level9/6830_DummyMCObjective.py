from torch.nn import Module
import torch
from torch import Tensor
from abc import ABC
from abc import abstractmethod


class AcquisitionObjective(Module, ABC):
    """Abstract base class for objectives."""
    ...


class MCAcquisitionObjective(AcquisitionObjective):
    """Abstract base class for MC-based objectives."""

    @abstractmethod
    def forward(self, samples: 'Tensor') ->Tensor:
        """Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
            values (assuming maximization).

        This method is usually not called directly, but via the objectives

        Example:
            >>> # `__call__` method:
            >>> samples = sampler(posterior)
            >>> outcome = mc_obj(samples)
        """
        pass


class DummyMCObjective(MCAcquisitionObjective):

    def forward(self, samples: 'Tensor') ->Tensor:
        return samples.sum(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
