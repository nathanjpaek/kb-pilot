import torch
import numpy as np
import torch.cuda
import torch.fft


def diagonal_quantize_function(x, bit, phase_noise_std=0, random_state=None,
    gradient_clip=False):


    class DiagonalQuantizeFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            S_scale = x.abs().max(dim=-1, keepdim=True)[0]
            x = (x / S_scale).acos()
            ratio = np.pi / (2 ** bit - 1)
            x.div_(ratio).round_().mul_(ratio)
            if phase_noise_std > 1e-05:
                noise = gen_gaussian_noise(x, noise_mean=0, noise_std=
                    phase_noise_std, trunc_range=[-2 * phase_noise_std, 2 *
                    phase_noise_std], random_state=random_state)
                x.add_(noise)
            x.cos_().mul_(S_scale)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            if gradient_clip:
                grad_input = grad_output.clamp(-1, 1)
            else:
                grad_input = grad_output.clone()
            return grad_input
    return DiagonalQuantizeFunction.apply(x)


class DiagonalQuantizer(torch.nn.Module):

    def __init__(self, bit, phase_noise_std=0.0, random_state=None, device=
        torch.device('cuda')):
        """2021/02/18: New phase quantizer for Sigma matrix in MZI-ONN. Gaussian phase noise is supported. All singular values are normalized by a TIA gain (S_scale), the normalized singular values will be achieved by cos(phi), phi will have [0, pi] uniform quantization.
        We do not consider real MZI implementation, thus voltage quantization and gamma noises are not supported.
        Args:
            bit (int): bitwidth for phase quantization.
            phase_noise_std (float, optional): Std dev for Gaussian phase noises. Defaults to 0.
            random_state (int, optional): random_state to control random noise injection. Defaults to None.
            device (torch.Device, optional): torch.Device. Defaults to torch.device("cuda").
        """
        super().__init__()
        self.bit = bit
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state
        self.device = device

    def set_phase_noise_std(self, phase_noise_std=0, random_state=None):
        self.phase_noise_std = phase_noise_std
        self.random_state = random_state

    def forward(self, x):
        x = diagonal_quantize_function(x, self.bit, self.phase_noise_std,
            self.random_state, gradient_clip=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'bit': 4}]
