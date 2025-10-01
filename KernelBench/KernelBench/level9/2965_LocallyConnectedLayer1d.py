import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class LocallyConnectedLayer1d(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=
        True, bias=False):
        """
      Defines one locally connected layer for one dimensional vector input.
      NOTE: This model only takes one-dimensional inputs. Batched inputs are also fine.

      input_dim: column size of weight matrix
      output_dim: row size of weight matrix
      kernel_size: number of local connections per parameter
      stride: number of strides of local connections, CANNOT BE ZERO
      padding: whether or not to zero pad
      bias: whether or not to have a bias term
      """
        super(LocallyConnectedLayer1d, self).__init__()
        initrange = 0.1
        self.weight = nn.Parameter(torch.empty(output_dim, kernel_size))
        torch.nn.init.uniform_(self.weight, -initrange, initrange)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is True:
            pad_size = stride * (output_dim - 1) + kernel_size - input_dim
            self.input_dim = input_dim + pad_size
            self.pad = True
        else:
            resulting_dim = (input_dim - kernel_size) / stride + 1
            self.extra_dim = output_dim - resulting_dim
            self.pad = False

    def forward(self, x):
        k = self.kernel_size
        s = self.stride
        instance_dim = len(x.size()) - 1
        if self.pad:
            pad_size = self.input_dim - x.size()[instance_dim]
            if pad_size >= 0:
                pad = 0, pad_size
                x = F.pad(x, pad=pad)
                x = x.unfold(instance_dim, k, s)
            else:
                x = x.unfold(instance_dim, k, s)
                if instance_dim == 0:
                    x = x[:pad_size]
                else:
                    x = x[:, :pad_size, :]
        else:
            x = x.unfold(instance_dim, k, s)
            for i in self.extra_dim:
                if instance_dim == 0:
                    x = torch.cat((x, x[-1]), dim=instance_dim)
                else:
                    x = torch.cat((x, x[:, -1, :]), dim=instance_dim)
        out = torch.sum(x * self.weight, dim=instance_dim + 1)
        if self.bias is not None:
            out += self.bias
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'kernel_size': 4,
        'stride': 1}]
