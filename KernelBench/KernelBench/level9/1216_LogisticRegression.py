import torch
from torch import nn
import torch.utils.data


class LogisticRegression(nn.Module):

    def __init__(self, input_units: 'int', output_units: 'int'):
        super().__init__()
        self._weights = nn.Parameter(torch.randn((input_units, output_units
            )), requires_grad=True)
        self._bias = nn.Parameter(torch.zeros(output_units), requires_grad=True
            )

    @property
    def weights(self) ->nn.Parameter:
        return self._weights

    @property
    def bias(self) ->nn.Parameter:
        return self._bias

    def forward(self, inputs) ->torch.Tensor:
        _logits = inputs.mm(self.weights) + self.bias
        _probs = torch.softmax(_logits, dim=1)
        return _probs

    def get_loss(self, inputs, y_true):
        _outputs = self.forward(inputs)
        _logmul_outputs = torch.log(_outputs + 1e-13) * y_true
        _logsum = torch.sum(_logmul_outputs, dim=1)
        _logsum_mean = torch.mean(_logsum)
        return -_logsum_mean


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_units': 4, 'output_units': 4}]
