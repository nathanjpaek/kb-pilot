import torch
import torch.utils.data
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func


class Conv1dKeepLength(torch_nn.Conv1d):
    """ Wrapper for causal convolution
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is optional
    """

    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, causal=
        False, stride=1, groups=1, bias=True, tanh=True, pad_mode='constant'):
        super(Conv1dKeepLength, self).__init__(input_dim, output_dim,
            kernel_s, stride=stride, padding=0, dilation=dilation_s, groups
            =groups, bias=bias)
        self.pad_mode = pad_mode
        self.causal = causal
        if self.causal:
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le
        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()

    def forward(self, data):
        x = torch_nn_func.pad(data.permute(0, 2, 1).unsqueeze(2), (self.
            pad_le, self.pad_ri, 0, 0), mode=self.pad_mode).squeeze(2)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output.permute(0, 2, 1)


class MovingAverage(Conv1dKeepLength):
    """ Wrapper to define a moving average smoothing layer
    Note: MovingAverage can be implemented using TimeInvFIRFilter too.
          Here we define another Module dicrectly on Conv1DKeepLength
    """

    def __init__(self, feature_dim, window_len, causal=False, pad_mode=
        'replicate'):
        super(MovingAverage, self).__init__(feature_dim, feature_dim, 1,
            window_len, causal, groups=feature_dim, bias=False, tanh=False,
            pad_mode=pad_mode)
        torch_nn.init.constant_(self.weight, 1 / window_len)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, data):
        return super(MovingAverage, self).forward(data)


class UpSampleLayer(torch_nn.Module):
    """ Wrapper over up-sampling
    Input tensor: (batchsize=1, length, dim)
    Ouput tensor: (batchsize=1, length * up-sampling_factor, dim)
    """

    def __init__(self, feature_dim, up_sampling_factor, smoothing=False):
        super(UpSampleLayer, self).__init__()
        self.scale_factor = up_sampling_factor
        self.l_upsamp = torch_nn.Upsample(scale_factor=self.scale_factor)
        if smoothing:
            self.l_ave1 = MovingAverage(feature_dim, self.scale_factor)
            self.l_ave2 = MovingAverage(feature_dim, self.scale_factor)
        else:
            self.l_ave1 = torch_nn.Identity()
            self.l_ave2 = torch_nn.Identity()
        return

    def forward(self, x):
        up_sampled_data = self.l_upsamp(x.permute(0, 2, 1))
        return self.l_ave1(self.l_ave2(up_sampled_data.permute(0, 2, 1)))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4, 'up_sampling_factor': 4}]
