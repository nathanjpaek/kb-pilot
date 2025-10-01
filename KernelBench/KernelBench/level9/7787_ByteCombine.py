import math
import torch
import torch.nn as nn
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


class ByteCombine(nn.Module):

    def __init__(self, input_dim, output_dim, inner_dim=1024, **kwags):
        super().__init__()
        self.layer_1 = ReRegualizedLinearNACLayer(input_dim, inner_dim)
        self.layer_2 = ReRegualizedLinearNACLayer(inner_dim, output_dim)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def forward(self, input):
        return self.act(self.layer_2(self.act(self.layer_1(input))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
