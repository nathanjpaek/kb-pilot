import math
import torch
import torch.nn as nn


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # noqa
    """

    def __init__(self, in1_features: 'int', in2_features: 'int',
        out_features: 'int', bias: 'bool'=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, out_features,
            in2_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: 'torch.Tensor', input2: 'torch.Tensor'
        ) ->torch.Tensor:
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y

    def extra_repr(self) ->str:
        return ('in1_features={}, in2_features={}, out_features={}, bias={}'
            .format(self.in1_features, self.in2_features, self.out_features,
            self.bias is not None))


class Biaffine(nn.Module):

    def __init__(self, in1_features: 'int', in2_features: 'int',
        out_features: 'int'):
        super().__init__()
        self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1,
            out_features)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1: 'torch.Tensor', input2: 'torch.Tensor'
        ) ->torch.Tensor:
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)
            ], dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)
            ], dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}]
