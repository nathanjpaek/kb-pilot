import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class ZIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, _out, others = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = 1 / gama * (1 / gama) * (gama - input.abs()).clamp(min=0)
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):

    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class SeqToANNContainer(nn.Module):

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: 'torch.Tensor'):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class APLayer(nn.Module):

    def __init__(self, kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(nn.AvgPool2d(kernel_size))
        self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        x = self.act(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
