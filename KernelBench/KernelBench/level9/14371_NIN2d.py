import torch
import torch.nn as nn
from torch.nn import Parameter


def norm(p: 'torch.Tensor', dim: 'int'):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size
            )
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*
            output_size)
    else:
        return norm(p.transpose(0, dim), 0).transpose(0, dim)


class NIN2d(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(NIN2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = Parameter(torch.Tensor(out_features, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_v, mean=0.0, std=0.05)
        self.weight_g.data.copy_(norm(self.weight_v, 0))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def compute_weight(self):
        return self.weight_v * (self.weight_g / norm(self.weight_v, 0))

    def forward(self, input):
        weight = self.compute_weight()
        out = torch.einsum('...cij,oc->...oij', (input, weight))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            out = self(x)
            out_features, height, width = out.size()[-3:]
            assert out_features == self.out_features
            out = out.view(-1, out_features, height * width).transpose(1, 2)
            out = out.contiguous().view(-1, out_features)
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-06)
            self.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.bias is not None:
                mean = mean.view(out_features, 1, 1)
                inv_stdv = inv_stdv.view(out_features, 1, 1)
                self.bias.add_(-mean).mul_(inv_stdv)
            return self(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
