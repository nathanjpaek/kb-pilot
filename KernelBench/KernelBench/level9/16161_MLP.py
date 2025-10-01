import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.cuda
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class Butterfly(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: 'randn', 'ortho', or 'identity'. Whether the weight matrix should be initialized to
            from randn twiddle, or to be randomly orthogonal/unitary, or to be the identity matrix.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, bias=True, complex=False,
        increasing_stride=True, init='randn', nblocks=1):
        super().__init__()
        self.in_size = in_size
        log_n = int(math.ceil(math.log2(in_size)))
        self.log_n = log_n
        size = self.in_size_extended = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.in_size_extended))
        self.complex = complex
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype(
            ) if not self.complex else real_dtype_to_complex[torch.
            get_default_dtype()]
        twiddle_shape = self.nstacks, nblocks, log_n, size // 2, 2, 2
        assert init in ['randn', 'ortho', 'identity']
        self.init = init
        self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.twiddle._is_structured = True
        if self.complex:
            self.twiddle = nn.Parameter(view_as_real(self.twiddle))
            if self.bias is not None:
                self.bias = nn.Parameter(view_as_real(self.bias))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        twiddle = self.twiddle if not self.complex else view_as_complex(self
            .twiddle)
        if self.init == 'randn':
            scaling = 1.0 / math.sqrt(2)
            with torch.no_grad():
                twiddle.copy_(torch.randn(twiddle.shape, dtype=twiddle.
                    dtype) * scaling)
        elif self.init == 'ortho':
            twiddle_core_shape = twiddle.shape[:-2]
            if not self.complex:
                theta = torch.rand(twiddle_core_shape) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype
                    ) * 2 - 1
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((det * c, -det *
                        s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2))
            else:
                phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.rand((3,) + twiddle_core_shape
                    ) * math.pi * 2
                A = torch.exp(1.0j * (alpha + psi)) * c
                B = torch.exp(1.0j * (alpha + chi)) * s
                C = -torch.exp(1.0j * (alpha - chi)) * s
                D = torch.exp(1.0j * (alpha - psi)) * c
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((A, B), dim=-1),
                        torch.stack((C, D), dim=-1)), dim=-2))
        elif self.init == 'identity':
            twiddle_new = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1,
                1, 2, 2)
            twiddle_new = twiddle_new.expand(*twiddle.shape).contiguous()
            with torch.no_grad():
                twiddle.copy_(twiddle_new)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False
        ):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.in_size_extended in this case
        Return:
            output: (batch, *, out_size)
        """
        twiddle = self.twiddle if not self.complex else view_as_complex(self
            .twiddle)
        if not subtwiddle:
            output = self.pre_process(input)
        else:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            output = self.pre_process(input, padded_size=n)
            twiddle = twiddle[:, :, :log_n, :n // 2
                ] if self.increasing_stride else twiddle[:, :, -log_n:, :n // 2
                ]
        if conjugate and self.complex:
            twiddle = twiddle.conj()
        if not transpose:
            output = butterfly_multiply(twiddle, output, self.increasing_stride
                )
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.
                nblocks - 1) % 2 == 1)
            output = butterfly_multiply(twiddle, output, not
                last_increasing_stride)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))

    def pre_process(self, input, padded_size=None):
        if padded_size is None:
            padded_size = self.in_size_extended
        input_size = input.size(-1)
        output = input.reshape(-1, input_size)
        batch = output.shape[0]
        if input_size != padded_size:
            output = F.pad(output, (0, padded_size - input_size))
        output = output.unsqueeze(1).expand(batch, self.nstacks, padded_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * output.size(-1))
        out_size_extended = 1 << int(math.ceil(math.log2(output.size(-1))))
        if out_size != out_size_extended:
            output = output[:, :out_size]
        if self.bias is not None:
            bias = self.bias if not self.complex else view_as_complex(self.bias
                )
            output = output + bias[:out_size]
        return output.view(*input.size()[:-1], out_size)

    def extra_repr(self):
        s = (
            'in_size={}, out_size={}, bias={}, complex={}, increasing_stride={}, init={}, nblocks={}'
            .format(self.in_size, self.out_size, self.bias is not None,
            self.complex, self.increasing_stride, self.init, self.nblocks))
        return s


class MLP(nn.Module):

    def __init__(self, method='linear', **kwargs):
        super().__init__()
        if method == 'linear':

            def make_layer(name):
                return self.add_module(name, nn.Linear(1024, 1024, bias=True))
        elif method == 'butterfly':

            def make_layer(name):
                return self.add_module(name, Butterfly(1024, 1024, bias=
                    True, **kwargs))
        elif method == 'low-rank':

            def make_layer(name):
                return self.add_module(name, nn.Sequential(nn.Linear(1024,
                    kwargs['rank'], bias=False), nn.Linear(kwargs['rank'], 
                    1024, bias=True)))
        elif method == 'toeplitz':

            def make_layer(name):
                return self.add_module(name, sl.ToeplitzLikeC(layer_size=
                    1024, bias=True, **kwargs))
        else:
            assert False, f'method {method} not supported'
        make_layer('fc10')
        make_layer('fc11')
        make_layer('fc12')
        make_layer('fc2')
        make_layer('fc3')
        self.logits = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 3, 1024)
        x = self.fc10(x[:, 0, :]) + self.fc11(x[:, 1, :]) + self.fc12(x[:, 
            2, :])
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.logits(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 1024])]


def get_init_inputs():
    return [[], {}]
