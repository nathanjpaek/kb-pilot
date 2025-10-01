from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class RepresentationModule(nn.Module):

    def __init__(self, config, task_name, repr_size):
        super(RepresentationModule, self).__init__()
        self.config = config
        self.task_name = task_name
        self.repr_size = repr_size
        self.to_repr = nn.Linear(config.hidden_size, self.repr_size)

    def forward(self, input, input_mask=None, segment_ids=None, extra_args=
        None, **kwargs):
        logits = self.to_repr(input[:, 0])
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4), 'task_name': 4,
        'repr_size': 4}]
