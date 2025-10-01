import torch
import torch.nn as nn


class TemperatureScaleTrainer(nn.Module):

    def __init__(self):
        super(TemperatureScaleTrainer, self).__init__()
        self._temperature = nn.Parameter(torch.ones(1), requires_grad=True)
        if torch.cuda.is_available():
            self._temperature
            self

    def forward(self, logits: 'torch.Tensor'):
        expanded_temperature = self._temperature.unsqueeze(1).expand(logits
            .size(0), logits.size(1))
        if torch.cuda.is_available():
            expanded_temperature = expanded_temperature
            logits
        return logits / expanded_temperature

    def get_parameters(self):
        return self._temperature

    def get_temperature(self):
        return self._temperature.item()

    def set_temperature(self, t: 'float'):
        self._temperature = nn.Parameter(torch.tensor([t]), requires_grad=True)
        if torch.cuda.is_available():
            self._temperature


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
