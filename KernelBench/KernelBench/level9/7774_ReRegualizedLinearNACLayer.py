import math
import torch
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class ReRegualizedLinearNACLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)

    def forward(self, input, reuse=False):
        W = torch.clamp(self.W, -1, 1)
        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features,
            self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
