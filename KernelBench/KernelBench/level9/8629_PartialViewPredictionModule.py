from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class PartialViewPredictionModule(nn.Module):

    def __init__(self, config, task_name, n_classes, activate=True):
        super(PartialViewPredictionModule, self).__init__()
        self.config = config
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.activate = activate
        if activate:
            self.activation = nn.SELU()
        self.to_logits = nn.Linear(config.hidden_size, n_classes)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.
                initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input):
        projected = self.projection(input)
        if self.activate:
            projected = self.activation(projected)
        logits = self.to_logits(projected)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4), 'task_name': 4,
        'n_classes': 4}]
