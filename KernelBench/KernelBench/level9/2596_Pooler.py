import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms.functional as F
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import Parameter


class Pooler(nn.Module):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.dense = Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, sequence_index=0):
        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Linear(nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, skip_bias_add=False
        ):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.weight = Parameter(torch.empty(self.output_size, self.input_size))
        init.normal_(self.weight)
        if bias:
            self.bias = Parameter(torch.empty(self.output_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(input_, self.weight, bias)
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output

    def __repr__(self):
        return (
            f'Linear(in_features={self.input_size}, out_features={self.output_size}, '
             +
            f'bias={self.bias is not None}, skip_bias_add={self.skip_bias_add})'
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
