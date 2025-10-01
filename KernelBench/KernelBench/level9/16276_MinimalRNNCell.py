import torch
from torch import nn
from functools import partial


def get_initializer(name, activation):
    if activation in ['id', 'identity', 'linear', 'modrelu']:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    else:
        assert False, f'get_initializer: activation {activation} not supported'
    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=
            nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=
            nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        assert False, f'get_initializer: initializer type {name} not supported'
    return initializer


def Linear_(input_size, output_size, bias, init='normal', zero_bias_init=
    False, **kwargs):
    """ Returns a nn.Linear module with initialization options """
    l = nn.Linear(input_size, output_size, bias=bias, **kwargs)
    get_initializer(init, 'linear')(l.weight)
    if bias and zero_bias_init:
        nn.init.zeros_(l.bias)
    return l


def get_activation(activation, size):
    if activation == 'id':
        return nn.Identity()
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'relu':
        return torch.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'modrelu':
        return Modrelu(size)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented"
            .format(activation))


class Gate(nn.Module):
    """ Implements gating mechanisms.

    Mechanisms:
    N  - No gate
    G  - Standard sigmoid gate
    """

    def __init__(self, size, preact_ctor, preact_args, mechanism='N'):
        super().__init__()
        self.size = size
        self.mechanism = mechanism
        if self.mechanism == 'N':
            pass
        elif self.mechanism == 'G':
            self.W_g = preact_ctor(*preact_args)
        else:
            assert False, f'Gating type {self.mechanism} is not supported.'

    def forward(self, *inputs):
        if self.mechanism == 'N':
            return 1.0
        if self.mechanism == 'G':
            g_preact = self.W_g(*inputs)
            g = torch.sigmoid(g_preact)
        return g


class modrelu(nn.Module):

    def __init__(self, features):
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)
        return phase * magnitude


class CellBase(nn.Module):
    """ Abstract class for our recurrent cell interface.

    Passes input through
    """
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'name') and cls.name is not None:
            cls.registry[cls.name] = cls
    name = 'id'
    valid_keys = []

    def default_initializers(self):
        return {}

    def default_architecture(self):
        return {}

    def __init__(self, input_size, hidden_size, initializers=None,
        architecture=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.architecture = self.default_architecture()
        self.initializers = self.default_initializers()
        if initializers is not None:
            self.initializers.update(initializers)
            None
        if architecture is not None:
            self.architecture.update(architecture)
        assert set(self.initializers.keys()).issubset(self.valid_keys)
        assert set(self.architecture.keys()).issubset(self.valid_keys)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input, hidden):
        return input, input

    def default_state(self, input, batch_size=None):
        return input.new_zeros(input.size(0) if batch_size is None else
            batch_size, self.hidden_size, requires_grad=False)

    def output(self, h):
        return h

    def state_size(self):
        return self.hidden_size

    def output_size(self):
        return self.hidden_size

    def initial_state(self, trainable=False):
        """ Return initial state of the RNN
        This should not need to see the input as it should be batch size agnostic and automatically broadcasted

        # TODO Currently not used
        """
        if trainable:
            self.initial_state = torch.zeros(self.hidden_size,
                requires_grad=True)
        else:
            return torch.zeros(self.hidden_size, requires_grad=True)


class Modrelu(modrelu):

    def reset_parameters(self):
        self.b.data.uniform_(-0.0, 0.0)


class MinimalRNNCell(CellBase):
    name = 'mrnn'
    valid_keys = ['hx', 'bias']

    def default_initializers(self):
        return {'hx': 'xavier'}

    def default_architecture(self):
        return {'bias': True}

    def __init__(self, input_size, hidden_size, hidden_activation='tanh',
        orthogonal=False, ortho_args=None, zero_bias_init=False, **kwargs):
        self.hidden_activation = hidden_activation
        self.zero_bias_init = zero_bias_init
        super().__init__(input_size, hidden_size, **kwargs)

    def reset_parameters(self):
        self.W_hx = Linear_(self.input_size, self.hidden_size, bias=self.
            architecture['bias'], zero_bias_init=self.zero_bias_init)
        get_initializer(self.initializers['hx'], self.hidden_activation)(self
            .W_hx.weight)
        self.hidden_activation_fn = get_activation(self.hidden_activation,
            self.hidden_size)
        preact_ctor = Linear_
        preact_args = [self.input_size + self.hidden_size, self.hidden_size,
            self.architecture['bias']]
        self.W_g = Gate(self.hidden_size, preact_ctor, preact_args,
            mechanism='G')

    def forward(self, input, h):
        hidden_preact = self.W_hx(input)
        hidden = self.hidden_activation_fn(hidden_preact)
        hx = torch.cat((input, h), dim=-1)
        g = self.W_g(hx)
        h = (1.0 - g) * h + g * hidden
        return h, h


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
