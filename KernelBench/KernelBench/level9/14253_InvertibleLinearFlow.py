import torch
import numpy as np
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


class InvertibleLinearFlow(Flow):

    def __init__(self, channels):
        super(InvertibleLinearFlow, self).__init__()
        self.channels = channels
        w_init = np.linalg.qr(np.random.randn(channels, channels))[0].astype(np
            .float32)
        self.register_parameter('weight', nn.Parameter(torch.from_numpy(
            w_init)))

    def forward(self, inputs: 'torch.Tensor', inputs_lengths=None) ->Tuple[
        torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = torch.matmul(inputs, self.weight)
        logdet = torch.linalg.slogdet(self.weight.double())[1].float()
        if inputs_lengths is None:
            logdet = torch.ones(input_shape[0], device=inputs.device) * float(
                input_shape[1]) * logdet
        else:
            logdet = inputs_lengths.float() * logdet
        return outputs, logdet

    def inverse(self, inputs: 'torch.Tensor', inputs_lengths=None) ->Tuple[
        torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = torch.matmul(inputs, torch.linalg.inv(self.weight))
        logdet = torch.linalg.slogdet(torch.linalg.inv(self.weight.double()))[1
            ].float()
        if inputs_lengths is None:
            logdet = torch.ones(input_shape[0], device=inputs.device) * float(
                input_shape[1]) * logdet
        else:
            logdet = inputs_lengths.float() * logdet
        return outputs, logdet

    def init(self, inputs: 'torch.Tensor', inputs_lengths=None):
        return self.forward(inputs, inputs_lengths)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
