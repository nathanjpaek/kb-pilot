import collections
import scipy
import torch
import numpy as np
import torch.utils.data
import scipy.stats
import scipy.optimize


def sparsity_error(W):
    W_error = torch.min(torch.abs(W), torch.abs(1 - torch.abs(W)))
    return torch.max(W_error)


def nac_w_variance(r):
    """Calculates the variance of W.

    Asumming \\hat{w} and \\hat{m} are sampled from a uniform
    distribution with range [-r, r], this is the variance
    of w = tanh(\\hat{w})*sigmoid(\\hat{m}).
    """
    if r == 0:
        return 0
    else:
        return (1 - np.tanh(r) / r) * (r - np.tanh(r / 2)) * (1 / (2 * r))


def nac_w_optimal_r(fan_in, fan_out):
    """Computes the optimal Uniform[-r, r] given the fan

    This uses numerical optimization.
    TODO: consider if there is an algebraic solution.
    """
    fan = max(fan_in + fan_out, 5)
    r = scipy.optimize.bisect(lambda r: fan * nac_w_variance(r) - 2, 0, 10)
    return r


def nac_weight(w_hat, m_hat, mode='normal'):
    if mode == 'normal':
        return NACWeight.apply(w_hat, m_hat)
    elif mode == 'sign':
        return NACWeightSign.apply(w_hat, m_hat)
    elif mode == 'independent':
        return NACWeightIndependent.apply(w_hat, m_hat)


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


class RegualizerNAUZ:

    def __init__(self, zero=False):
        self.zero = zero
        self.stored_inputs = []

    def __call__(self, W):
        if self.zero:
            return 0
        x_mean = torch.mean(torch.cat(self.stored_inputs, dim=0), dim=0,
            keepdim=True)
        return torch.mean((1 - torch.abs(W)) * (0 - x_mean) ** 2)

    def append_input(self, x):
        if self.zero:
            return
        self.stored_inputs.append(x)

    def reset(self):
        if self.zero:
            return
        self.stored_inputs = []


class NACWeight(torch.autograd.Function):
    """Implements the NAC weight operator

    w = tanh(\\hat{w}) * sigmoid(\\hat{m})
    """

    @staticmethod
    def forward(ctx, w_hat, m_hat):
        tanh_w_hat = torch.tanh(w_hat)
        sigmoid_m_hat = torch.sigmoid(m_hat)
        ctx.save_for_backward(tanh_w_hat, sigmoid_m_hat)
        return tanh_w_hat * sigmoid_m_hat

    @staticmethod
    def backward(ctx, grad_output):
        tanh_w_hat, sigmoid_m_hat = ctx.saved_tensors
        return grad_output * (1 - tanh_w_hat * tanh_w_hat
            ) * sigmoid_m_hat, grad_output * tanh_w_hat * sigmoid_m_hat * (
            1 - sigmoid_m_hat)


class NACWeightIndependent(torch.autograd.Function):
    """Implements the NAC weight operator but with independent optimization.

    The optimiation of \\hat{w} is independent of \\hat{m} and vice versa.

    w = tanh(\\hat{w}) * sigmoid(\\hat{m})

    dL/d\\hat{w} = (dL/dw) (dw/d\\hat{w})
                = (dL/dw) (1 - tanh(\\hat{w})^2)

    dL/d\\hat{m} = (dL/dw) (dw/d\\hat{m})
                = (dL/dw) sigmoid(\\hat{m}) * (1 - sigmoid(\\hat{m}))
    """

    @staticmethod
    def forward(ctx, w_hat, m_hat):
        tanh_w_hat = torch.tanh(w_hat)
        sigmoid_m_hat = torch.sigmoid(m_hat)
        ctx.save_for_backward(tanh_w_hat, sigmoid_m_hat)
        return tanh_w_hat * sigmoid_m_hat

    @staticmethod
    def backward(ctx, grad_output):
        tanh_w_hat, sigmoid_m_hat = ctx.saved_tensors
        return grad_output * (1 - tanh_w_hat * tanh_w_hat
            ), grad_output * sigmoid_m_hat * (1 - sigmoid_m_hat)


class NACWeightSign(torch.autograd.Function):
    """Implements the NAC weight operator but with a hard gradient for \\hat{m}

    w = tanh(\\hat{w}) * sigmoid(\\hat{m})
    dL/d\\hat{m} = (dL/dw) (dw/d\\hat{m})
                = (dL/dw) * 0.1 * sign(\\hat{w}) * sigmoid(\\hat{m}) * (1 - sigmoid(\\hat{m}))
    """

    @staticmethod
    def forward(ctx, w_hat, m_hat):
        tanh_w_hat = torch.tanh(w_hat)
        sigmoid_m_hat = torch.sigmoid(m_hat)
        ctx.save_for_backward(w_hat, tanh_w_hat, sigmoid_m_hat)
        return tanh_w_hat * sigmoid_m_hat

    @staticmethod
    def backward(ctx, grad_output):
        w_hat, tanh_w_hat, sigmoid_m_hat = ctx.saved_tensors
        return grad_output * (1 - tanh_w_hat * tanh_w_hat
            ) * sigmoid_m_hat, grad_output * 0.1 * torch.sign(w_hat
            ) * sigmoid_m_hat * (1 - sigmoid_m_hat)


class NACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, regualizer_z=0, **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.W_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features)
            )
        self.M_hat = torch.nn.Parameter(torch.Tensor(out_features, in_features)
            )
        self.register_parameter('bias', None)
        self._regualizer_nau_z = RegualizerNAUZ(zero=regualizer_z == 0)

    def reset_parameters(self):
        r = nac_w_optimal_r(self.in_features, self.out_features)
        torch.nn.init.uniform_(self.W_hat, a=-r, b=r)
        torch.nn.init.uniform_(self.M_hat, a=-r, b=r)

    def optimize(self, loss):
        self._regualizer_nau_z.reset()

    def regualizer(self):
        W = nac_weight(self.W_hat, self.M_hat, mode='normal')
        return super().regualizer({'z': self._regualizer_nau_z(W)})

    def forward(self, x, reuse=False):
        if self.allow_random:
            self._regualizer_nau_z.append_input(x)
        W = nac_weight(self.W_hat, self.M_hat, mode='normal')
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W),
            verbose_only=False)
        return torch.nn.functional.linear(x, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features,
            self.out_features)


class IndependentNACLayer(NACLayer):

    def forward(self, input, reuse=False):
        W = nac_weight(self.W_hat, self.M_hat, mode='independent')
        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W),
            verbose_only=False)
        return torch.nn.functional.linear(input, W, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
