import collections
import math
import torch
import torch.utils.data


def sparsity_error(W):
    W_error = torch.min(torch.abs(W), torch.abs(1 - torch.abs(W)))
    return torch.max(W_error)


class SummaryWriterNamespaceNoLoggingScope:

    def __init__(self, writer):
        self._writer = writer

    def __enter__(self):
        self._writer._logging_enabled = False

    def __exit__(self, type, value, traceback):
        self._writer._logging_enabled = True
        return False


class DummySummaryWriter:

    def __init__(self, **kwargs):
        self._logging_enabled = False
        pass

    def add_scalar(self, name, value, verbose_only=True):
        pass

    def add_summary(self, name, tensor, verbose_only=True):
        pass

    def add_histogram(self, name, tensor, verbose_only=True):
        pass

    def add_tensor(self, name, tensor, verbose_only=True):
        pass

    def print(self, name, tensor, verbose_only=True):
        pass

    def namespace(self, name):
        return self

    def every(self, epoch_interval):
        return self

    def verbose(self, verbose):
        return self

    def no_logging(self):
        return SummaryWriterNamespaceNoLoggingScope(self)


class NoRandomScope:

    def __init__(self, module):
        self._module = module

    def __enter__(self):
        self._module._disable_random()

    def __exit__(self, type, value, traceback):
        self._module._enable_random()
        return False


class ExtendedTorchModule(torch.nn.Module):

    def __init__(self, default_name, *args, writer=None, name=None, **kwargs):
        super().__init__()
        if writer is None:
            writer = DummySummaryWriter()
        self.writer = writer.namespace(default_name if name is None else name)
        self.allow_random = True

    def set_parameter(self, name, value):
        parameter = getattr(self, name, None)
        if isinstance(parameter, torch.nn.Parameter):
            parameter.fill_(value)
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.set_parameter(name, value)

    def regualizer(self, merge_in=None):
        regualizers = collections.defaultdict(int)
        if merge_in is not None:
            for key, value in merge_in.items():
                self.writer.add_scalar(f'regualizer/{key}', value)
                regualizers[key] += value
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                for key, value in module.regualizer().items():
                    regualizers[key] += value
        return regualizers

    def optimize(self, loss):
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.optimize(loss)

    def log_gradients(self):
        for name, parameter in self.named_parameters(recurse=False):
            if parameter.requires_grad:
                gradient, *_ = parameter.grad.data
                self.writer.add_summary(f'{name}/grad', gradient)
                self.writer.add_histogram(f'{name}/grad', gradient)
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.log_gradients()

    def no_internal_logging(self):
        return self.writer.no_logging()

    def _disable_random(self):
        self.allow_random = False
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module._disable_random()

    def _enable_random(self):
        self.allow_random = True
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module._enable_random()

    def no_random(self):
        return NoRandomScope(self)


class Regualizer:

    def __init__(self, support='nac', type='bias', shape='squared', zero=
        False, zero_epsilon=0):
        super()
        self.zero_epsilon = 0
        if zero:
            self.fn = self._zero
        else:
            identifier = '_'.join(['', support, type, shape])
            self.fn = getattr(self, identifier)

    def __call__(self, W):
        return self.fn(W)

    def _zero(self, W):
        return 0

    def _mnac_bias_linear(self, W):
        return torch.mean(torch.min(torch.abs(W - self.zero_epsilon), torch
            .abs(1 - W)))

    def _mnac_bias_squared(self, W):
        return torch.mean((W - self.zero_epsilon) ** 2 * (1 - W) ** 2)

    def _mnac_oob_linear(self, W):
        return torch.mean(torch.relu(torch.abs(W - 0.5 - self.zero_epsilon) -
            0.5 + self.zero_epsilon))

    def _mnac_oob_squared(self, W):
        return torch.mean(torch.relu(torch.abs(W - 0.5 - self.zero_epsilon) -
            0.5 + self.zero_epsilon) ** 2)

    def _nac_bias_linear(self, W):
        W_abs = torch.abs(W)
        return torch.mean(torch.min(W_abs, torch.abs(1 - W_abs)))

    def _nac_bias_squared(self, W):
        return torch.mean(W ** 2 * (1 - torch.abs(W)) ** 2)

    def _nac_oob_linear(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1))

    def _nac_oob_squared(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1) ** 2)


class RegualizerNMUZ:

    def __init__(self, zero=False):
        self.zero = zero
        self.stored_inputs = []

    def __call__(self, W):
        if self.zero:
            return 0
        x_mean = torch.mean(torch.cat(self.stored_inputs, dim=0), dim=0,
            keepdim=True)
        return torch.mean((1 - W) * (1 - x_mean) ** 2)

    def append_input(self, x):
        if self.zero:
            return
        self.stored_inputs.append(x)

    def reset(self):
        if self.zero:
            return
        self.stored_inputs = []


class ReRegualizedLinearPosNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, nac_oob='regualized',
        regualizer_shape='squared', mnac_epsilon=0, mnac_normalized=False,
        regualizer_z=0, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized
        self.mnac_epsilon = mnac_epsilon
        self.nac_oob = nac_oob
        self._regualizer_bias = Regualizer(support='mnac', type='bias',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon)
        self._regualizer_oob = Regualizer(support='mnac', type='oob', shape
            =regualizer_shape, zero_epsilon=mnac_epsilon, zero=self.nac_oob ==
            'clip')
        self._regualizer_nmu_z = RegualizerNMUZ(zero=regualizer_z == 0)
        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)
        self._regualizer_nmu_z.reset()

    def optimize(self, loss):
        self._regualizer_nmu_z.reset()
        if self.nac_oob == 'clip':
            self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return super().regualizer({'W': self._regualizer_bias(self.W), 'z':
            self._regualizer_nmu_z(self.W), 'W-OOB': self._regualizer_oob(
            self.W)})

    def forward(self, x, reuse=False):
        if self.allow_random:
            self._regualizer_nmu_z.append_input(x)
        W = torch.clamp(self.W, 0.0 + self.mnac_epsilon, 1.0
            ) if self.nac_oob == 'regualized' else self.W
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W),
            verbose_only=False)
        return torch.nn.functional.linear(x, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features,
            self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
