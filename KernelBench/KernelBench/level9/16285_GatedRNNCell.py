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


class Parametrization(nn.Module):
    """
    Implements the parametrization of a manifold in terms of a Euclidean space

    It gives the parametrized matrix through the attribute `B`

    To use it, subclass it and implement the method `retraction` and the method `forward` (and optionally `project`). See the documentation in these methods for details

    You can find an example in the file `orthogonal.py` where we implement the Orthogonal class to optimize over the Stiefel manifold using an arbitrary retraction
    """

    def __init__(self, A, base, mode):
        """
        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.
        """
        super(Parametrization, self).__init__()
        assert mode == 'static' or isinstance(mode, tuple) and len(mode
            ) == 3 and mode[0] == 'dynamic'
        self.A = nn.Parameter(A)
        self.register_buffer('_B', None)
        self.register_buffer('base', base)
        if mode == 'static':
            self.mode = mode
        else:
            self.mode = mode[0]
            self.K = mode[1]
            self.M = mode[2]
            self.k = 0
            self.m = 0

        def hook(grad):
            nonlocal self
            self._B = None
        self.A.register_hook(hook)

    def rebase(self):
        with torch.no_grad():
            self.base.data.copy_(self._B.data)
            self.A.data.zero_()

    @property
    def B(self):
        not_B = self._B is None
        if not_B or not self._B.grad_fn and torch.is_grad_enabled():
            self._B = self.retraction(self.A, self.base)
            self._B.requires_grad_()
            self._B.retain_grad()
            if self.mode == 'dynamic' and not_B:
                if self.k == 0:
                    self.rebase()
                    self.m = (self.m + 1) % self.M
                    if self.m == 0 and hasattr(self, 'project'):
                        with torch.no_grad():
                            self.base = self.project(self.base)
                if self.K != 'infty':
                    self.k = (self.k + 1) % self.K
                elif self.k == 0:
                    self.k = 1
        return self._B

    def retraction(self, A, base):
        """
        It computes r_{base}(A).
        Notice that A will not always be in the tangent space of our manifold
          For this reason, we first have to use A to parametrize the tangent space,
          and then compute the retraction
        When dealing with Lie groups, raw_A is always projected into the Lie algebra, as an optimization (cf. Section E in the paper)
        """
        raise NotImplementedError

    def project(self, base):
        """
        This method is OPTIONAL
        It returns the projected base back into the manifold
        """
        raise NotImplementedError

    def forward(self, input):
        """
        It uses the attribute self.B to implement the layer itself (e.g. Linear, CNN, ...)
        """
        raise NotImplementedError


class Orthogonal(Parametrization):
    """ Class that implements optimization restricted to the Stiefel manifold """

    def __init__(self, input_size, output_size, initializer_skew, mode, param):
        """
        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.

        param: A parametrization of in terms of skew-symmetyric matrices
        """
        max_size = max(input_size, output_size)
        A = torch.empty(max_size, max_size)
        base = torch.empty(input_size, output_size)
        super(Orthogonal, self).__init__(A, base, mode)
        self.input_size = input_size
        self.output_size = output_size
        self.param = param
        self.init_A = initializer_skew
        self.init_base = nn.init.eye_
        self.reset_parameters()

    def reset_parameters(self):
        self.init_A(self.A)
        self.init_base(self.base)

    def forward(self, input):
        return input.matmul(self.B)

    def retraction(self, A, base):
        A = A.triu(diagonal=1)
        A = A - A.t()
        B = base.mm(self.param(A))
        if self.input_size != self.output_size:
            B = B[:self.input_size, :self.output_size]
        return B

    def project(self, base):
        try:
            U, _, V = torch.svd(base, some=True)
            return U.mm(V.t())
        except RuntimeError:
            x = base
            if base.size(0) < base.size(1):
                x = base.t()
            ret = torch.qr(x, some=True).Q
            if base.size(0) < base.size(1):
                ret = ret.t()
            return ret


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


class OrthogonalLinear(Orthogonal):

    def __init__(self, input_size, output_size, method='exprnn', init=
        'cayley', K=100):
        """ Wrapper around expRNN's Orthogonal class taking care of parameter names """
        if method == 'exprnn':
            mode = 'static'
            param = 'expm'
        elif method == 'dtriv':
            mode = 'dynamic', ortho_args['K'], 100
            param = 'expm'
        elif method == 'cayley':
            mode = 'static'
            param = 'cayley'
        else:
            assert False, f'OrthogonalLinear: orthogonal method {method} not supported'
        param = param_name_to_param[param]
        init_A = init_name_to_init[init]
        super().__init__(input_size, output_size, init_A, mode, param)


class RNNCell(CellBase):
    name = 'rnn'
    valid_keys = ['hx', 'hh', 'bias']

    def default_initializers(self):
        return {'hx': 'xavier', 'hh': 'xavier'}

    def default_architecture(self):
        return {'bias': True}

    def __init__(self, input_size, hidden_size, hidden_activation='tanh',
        orthogonal=False, ortho_args=None, zero_bias_init=False, **kwargs):
        self.hidden_activation = hidden_activation
        self.orthogonal = orthogonal
        self.ortho_args = ortho_args
        self.zero_bias_init = zero_bias_init
        super().__init__(input_size, hidden_size, **kwargs)

    def reset_parameters(self):
        self.W_hx = Linear_(self.input_size, self.hidden_size, bias=self.
            architecture['bias'], zero_bias_init=self.zero_bias_init)
        get_initializer(self.initializers['hx'], self.hidden_activation)(self
            .W_hx.weight)
        self.hidden_activation_fn = get_activation(self.hidden_activation,
            self.hidden_size)
        self.reset_hidden_to_hidden()

    def reset_hidden_to_hidden(self):
        if self.orthogonal:
            if self.ortho_args is None:
                self.ortho_args = {}
            self.ortho_args['input_size'] = self.hidden_size
            self.ortho_args['output_size'] = self.hidden_size
            self.W_hh = OrthogonalLinear(**self.ortho_args)
        else:
            self.W_hh = nn.Linear(self.hidden_size, self.hidden_size, bias=
                self.architecture['bias'])
            get_initializer(self.initializers['hh'], self.hidden_activation)(
                self.W_hh.weight)

    def forward(self, input, h):
        hidden_preact = self.W_hx(input) + self.W_hh(h)
        hidden = self.hidden_activation_fn(hidden_preact)
        return hidden, hidden


class GatedRNNCell(RNNCell):
    name = 'gru'

    def __init__(self, input_size, hidden_size, gate='G', reset='N', **kwargs):
        self.gate = gate
        self.reset = reset
        super().__init__(input_size, hidden_size, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        preact_ctor = Linear_
        preact_args = [self.input_size + self.hidden_size, self.hidden_size,
            self.architecture['bias']]
        self.W_g = Gate(self.hidden_size, preact_ctor, preact_args,
            mechanism=self.gate)
        self.W_reset = Gate(self.hidden_size, preact_ctor, preact_args,
            mechanism=self.reset)

    def forward(self, input, h):
        hx = torch.cat((input, h), dim=-1)
        reset = self.W_reset(hx)
        _, update = super().forward(input, reset * h)
        g = self.W_g(hx)
        h = (1.0 - g) * h + g * update
        return h, h


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
