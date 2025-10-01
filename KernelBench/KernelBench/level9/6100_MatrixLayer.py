import torch
import torch.nn as nn


class ActionPool(nn.Module):
    """
    Basic pooling operations. 
    """

    def __init__(self, axis, function='mean', expand=True):
        super(ActionPool, self).__init__()
        self.expand = expand
        self._function_name = function
        self._axis_name = axis
        if isinstance(axis, str):
            self.dim = {'row': 0, 'column': 1, 'both': None}[axis]
        else:
            self.dim = axis
        if function == 'max':
            self.function = lambda x, dim: torch.max(x, dim=dim)[0]
        elif function == 'sum':
            self.function = lambda x, dim: torch.sum(x, dim=dim)
        elif function == 'mean':
            self.function = torch.mean
        else:
            raise ValueError('Unrecognised function: %s' % function)

    def forward(self, input):
        if self.dim is None:
            n, p, i, j = input.size()
            input = input.contiguous()
            reshaped = input.view(n, p, i * j)
            output = self.function(reshaped, dim=2)
            if self.expand:
                output = output.unsqueeze(2).unsqueeze(2)
        else:
            output = self.function(input, dim=self.dim + 2)
            if self.expand:
                output = output.unsqueeze(self.dim + 2)
        if self.expand:
            return output.expand_as(input)
        else:
            return output


class MatrixLinear(nn.Linear):
    """
    Matrix-based linear feed-forward layer. Think of it as the
    matrix analog to a feed-forward layer in an MLP.
    """

    def forward(self, input):
        n, p, i, j = input.size()
        input.permute(0, 2, 3, 1)
        state = input.view((n * i * j, p))
        output = super(MatrixLinear, self).forward(state)
        output = output.view((n, i, j, self.out_features)).permute(0, 3, 1, 2)
        return output


class MatrixLayer(nn.Module):
    """
    Set layers are linear layers with pooling. Pooling operations
    are defined above.
    """

    def __init__(self, in_features, out_features, pooling='max', axes=[
        'row', 'column', 'both'], name='', debug=False, dropout=0.0):
        super(MatrixLayer, self).__init__()
        pool = []
        for axis in axes:
            pool.append(ActionPool(axis, pooling, expand=True))
        self.pool = pool
        self.dropout = dropout
        self.linear = MatrixLinear(in_features * (1 + len(pool)), out_features)
        if dropout > 0.0:
            self.dropoutlayer = nn.Dropout2d(dropout)
        self.name = name
        self.debug = debug

    def forward(self, input):
        pooled = [p(input) for p in self.pool]
        state = torch.cat([input] + pooled, dim=1)
        if self.dropout > 0.0:
            state = self.dropoutlayer(state)
        if self.debug:
            None
        return self.linear(state)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
