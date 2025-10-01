import torch
import torch.nn as nn
from typing import Tuple


class Flow(nn.Module):

    def __init__(self):
        super(Flow, self).__init__()

    def forward(self, *inputs, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            *inputs: input [batch, *input_size]
        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\\partial output / \\partial input`
        """
        raise NotImplementedError

    def inverse(self, *inputs, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            *input: input [batch, *input_size]
        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\\partial output / \\partial input`
        """
        raise NotImplementedError

    def init(self, *inputs, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Initiate the weights according to the initial input data
        :param inputs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class ActNormFlow(Flow):

    def __init__(self, channels):
        super(ActNormFlow, self).__init__()
        self.channels = channels
        self.register_parameter('log_scale', nn.Parameter(torch.normal(0.0,
            0.05, [self.channels])))
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.
            channels)))

    def forward(self, inputs: 'torch.Tensor', input_lengths=None) ->Tuple[
        torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = inputs * torch.exp(self.log_scale) + self.bias
        logdet = torch.sum(self.log_scale)
        if input_lengths is None:
            logdet = torch.ones(input_shape[0], device=inputs.device) * float(
                input_shape[1]) * logdet
        else:
            logdet = input_lengths.float() * logdet
        return outputs, logdet

    def inverse(self, inputs: 'torch.Tensor', input_lengths=None, epsilon=1e-08
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = (inputs - self.bias) / (torch.exp(self.log_scale) + epsilon)
        logdet = -torch.sum(self.log_scale)
        if input_lengths is None:
            logdet = torch.ones(input_shape[0], device=inputs.device) * float(
                input_shape[1]) * logdet
        else:
            logdet = input_lengths.float() * logdet
        return outputs, logdet

    def init(self, inputs: 'torch.Tensor', input_lengths=None, init_scale=
        1.0, epsilon=1e-08):
        _mean = torch.mean(inputs.view(-1, self.channels), dim=0)
        _std = torch.std(inputs.view(-1, self.channels), dim=0)
        self.log_scale.copy_(torch.log(init_scale / (_std + epsilon)))
        self.bias.copy_(-_mean / (_std + epsilon))
        return self.forward(inputs, input_lengths)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
