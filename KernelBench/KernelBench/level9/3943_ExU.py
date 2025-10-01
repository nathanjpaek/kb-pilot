import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ExU(torch.nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int') ->None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(self, inputs: 'torch.Tensor', n: 'int'=1) ->torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))
        output = F.relu(output)
        output = torch.clamp(output, 0, n)
        return output

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
