import torch
import torch.nn as nn


class HeatedUpScalar(nn.Module):

    def __init__(self, first_value, last_value, nb_steps, scope='task', **
        kwargs):
        super().__init__()
        self.scope = scope
        self.first_value = first_value
        self.step = (max(first_value, last_value) - min(first_value,
            last_value)) / (nb_steps - 1)
        if first_value > last_value:
            self._factor = -1
        else:
            self._factor = 1
        self._increment = 0
        None

    def on_task_end(self):
        if self.scope == 'task':
            self._increment += 1
        None

    def on_epoch_end(self):
        if self.scope == 'epoch':
            self._increment += 1

    @property
    def factor(self):
        return self.first_value + self._factor * self._increment * self.step

    def forward(self, inputs):
        return self.factor * inputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'first_value': 4, 'last_value': 4, 'nb_steps': 4}]
