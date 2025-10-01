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


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'dilation_s': 1,
        'kernel_s': 4}]
