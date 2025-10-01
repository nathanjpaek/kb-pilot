import torch
from abc import abstractmethod
from torch import nn
from torch.nn.utils import weight_norm


class HyperNet(nn.Module):
    """This module is responsible for taking the losses from all tasks and return a single loss term.
    We can think of this as our learnable loss criterion

    """

    def __init__(self, main_task, input_dim):
        super().__init__()
        self.main_task = main_task
        self.input_dim = input_dim

    def forward(self, losses, outputs=None, labels=None, data=None):
        """

        :param losses: losses form each task. This should be a tensor of size (batch_size, self.input_dim)
        :param outputs: Optional. Parameters model output.
        :param labels: Optional. Target.
        :param data: Optiona. Parameters model input.
        :return:
        """
        pass

    def _init_weights(self):
        pass

    def get_weights(self):
        """

        :return: list of model parameters
        """
        return list(self.parameters())


class MonoHyperNet(HyperNet):
    """Monotonic Hypernets

    """

    def __init__(self, main_task, input_dim, clamp_bias=False):
        super().__init__(main_task=main_task, input_dim=input_dim)
        self.clamp_bias = clamp_bias

    def get_weights(self):
        """

        :return: list of model parameters
        """
        return list(self.parameters())

    @abstractmethod
    def clamp(self):
        pass


class MonoLinearHyperNet(MonoHyperNet):
    """Linear weights, e.g. \\sum_j lpha_j * l_j

    """

    def __init__(self, main_task, input_dim, skip_connection=False,
        clamp_bias=False, init_value=1.0, weight_normalization=True):
        super().__init__(main_task=main_task, input_dim=main_task,
            clamp_bias=clamp_bias)
        self.init_value = init_value
        self.skip_connection = skip_connection
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self._init_weights()
        self.weight_normalization = weight_normalization
        if self.weight_normalization:
            self.linear = weight_norm(self.linear)

    def _init_weights(self):
        self.linear.weight = nn.init.constant_(self.linear.weight, self.
            init_value)

    def forward(self, losses, outputs=None, labels=None, data=None):
        loss = self.linear(losses).mean()
        if self.skip_connection:
            loss += losses[:, self.main_task].mean()
        return loss

    def clamp(self):
        """make sure parameters are non-negative

        """
        if self.weight_normalization:
            self.linear.weight_v.data.clamp_(0)
            self.linear.weight_g.data.clamp_(0)
        else:
            self.linear.weight.data.clamp_(0)
        if self.linear.bias is not None and self.clamp_bias:
            self.linear.bias.data.clamp_(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'main_task': 4, 'input_dim': 4}]
