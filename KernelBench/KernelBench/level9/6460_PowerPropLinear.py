import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerPropLinear(nn.Linear):
    """Powerpropagation Linear module."""

    def __init__(self, in_features, out_fetaures, alpha, bias=True, *args,
        **kwargs):
        self._alpha = alpha
        super(PowerPropLinear, self).__init__(in_features, out_fetaures,
            bias, *args, **kwargs)

    def reset_parameters(self):
        super(PowerPropLinear, self).reset_parameters()
        with torch.no_grad():
            weight = self.weight
            weight_modified = torch.sign(weight) * torch.pow(torch.abs(
                weight), 1.0 / self._alpha)
            self.weight.copy_(weight_modified)

    def get_weights(self):
        return torch.sign(self.weight) * torch.pow(torch.abs(self.weight),
            self._alpha)

    def forward(self, inputs, mask=None):
        params = self.weight * torch.pow(torch.abs(self.weight), self.
            _alpha - 1)
        if mask is not None:
            params *= mask
        outputs = F.linear(inputs, params, self.bias)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_fetaures': 4, 'alpha': 4}]
