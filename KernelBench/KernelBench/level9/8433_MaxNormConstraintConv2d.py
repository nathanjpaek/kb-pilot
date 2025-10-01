import torch
import torch.nn as nn


class MaxNormConstraintConv2d(nn.Conv2d):

    def __init__(self, *args, max_norm_value=1, norm_axis=2, **kwargs):
        self.max_norm_value = max_norm_value
        self.norm_axis = norm_axis
        super().__init__(*args, **kwargs)

    def forward(self, input):
        self.weight.data = self._max_norm(self.weight.data)
        return super().forward(input)

    def _max_norm(self, w):
        with torch.no_grad():
            norms = torch.sqrt(torch.sum(torch.square(w), dim=self.
                norm_axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_norm_value)
            w *= desired / norms
        return w


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
