import torch
from typing import Dict
from typing import Tuple
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class Flow(nn.Module):
    """
    Normalizing Flow base class
    """
    _registry = dict()

    def __init__(self, inverse):
        super(Flow, self).__init__()
        self.inverse = inverse

    def forward(self, *inputs, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\\partial output / \\partial input`
        """
        raise NotImplementedError

    def backward(self, *inputs, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\\partial output / \\partial input`
        """
        raise NotImplementedError

    def init(self, *input, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def fwdpass(self, x: 'torch.Tensor', *h, init=False, init_scale=1.0, **
        kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor
                The random variable before flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\\partial y / \\partial x`
            Then the density :math:`\\log(p(y)) = \\log(p(x)) - logdet`

        """
        if self.inverse:
            if init:
                raise RuntimeError(
                    'inverse flow shold be initialized with backward pass')
            else:
                return self.backward(x, *h, **kwargs)
        elif init:
            return self.init(x, *h, init_scale=init_scale, **kwargs)
        else:
            return self.forward(x, *h, **kwargs)

    def bwdpass(self, y: 'torch.Tensor', *h, init=False, init_scale=1.0, **
        kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            y: Tensor
                The random variable after flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: x: Tensor, logdet: Tensor
            x, the random variable before flow
            logdet, the log determinant of :math:`\\partial x / \\partial y`
            Then the density :math:`\\log(p(y)) = \\log(p(x)) + logdet`

        """
        if self.inverse:
            if init:
                return self.init(y, *h, init_scale=init_scale, **kwargs)
            else:
                return self.forward(y, *h, **kwargs)
        elif init:
            raise RuntimeError(
                'forward flow should be initialzed with forward pass')
        else:
            return self.backward(y, *h, **kwargs)

    @classmethod
    def register(cls, name: 'str'):
        Flow._registry[name] = cls

    @classmethod
    def by_name(cls, name: 'str'):
        return Flow._registry[name]

    @classmethod
    def from_params(cls, params: 'Dict'):
        raise NotImplementedError


class InvertibleLinearFlow(Flow):

    def __init__(self, in_features, inverse=False):
        super(InvertibleLinearFlow, self).__init__(inverse)
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    def forward(self, input: 'torch.Tensor', mask: 'torch.Tensor') ->Tuple[
        torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\\partial output / \\partial input`

        """
        dim = input.dim()
        out = F.linear(input, self.weight)
        _, logdet = torch.linalg.slogdet(self.weight)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    def backward(self, input: 'torch.Tensor', mask: 'torch.Tensor') ->Tuple[
        torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\\partial output / \\partial input`

        """
        dim = input.dim()
        out = F.linear(input, self.weight_inv)
        _, logdet = torch.linalg.slogdet(self.weight_inv)
        if dim > 2:
            num = mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    def init(self, data: 'torch.Tensor', mask: 'torch.Tensor', init_scale=1.0
        ) ->Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.
            in_features)

    @classmethod
    def from_params(cls, params: 'Dict') ->'InvertibleLinearFlow':
        return InvertibleLinearFlow(**params)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
