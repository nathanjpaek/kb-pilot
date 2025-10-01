import collections
import math
import torch
import torch.utils.data


def sparsity_error(W):
    W_error = torch.min(torch.abs(W), torch.abs(1 - torch.abs(W)))
    return torch.max(W_error)


def mnac(x, W, mode='prod'):
    out_size, in_size = W.size()
    x = x.view(x.size()[0], in_size, 1)
    W = W.t().view(1, in_size, out_size)
    if mode == 'prod':
        return torch.prod(x * W + 1 - W, -2)
    elif mode == 'exp-log':
        return torch.exp(torch.sum(torch.log(x * W + 1 - W), -2))
    elif mode == 'no-idendity':
        return torch.prod(x * W, -2)
    else:
        raise ValueError(f'mnac mode "{mode}" is not implemented')


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


class MNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features)
            )
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = math.sqrt(3.0) * std
        torch.nn.init.uniform_(self.W_hat, -r, r)

    def forward(self, x, reuse=False):
        W = torch.sigmoid(self.W_hat)
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W),
            verbose_only=False)
        return mnac(x, W)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features,
            self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 1])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
