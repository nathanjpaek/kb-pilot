import torch
import numpy as np
import torch.nn as nn


def cylindricalToPolarConversion(input1, input2=None):
    if input2 is None:
        """input1 is tensor of [B,C,H,W,D,2] contains both real and imaginary channels
         in the last dims"""
        ndims = input1.ndimension()
        real_input = input1.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        imag_input = input1.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)
        mag = (real_input ** 2 + imag_input ** 2) ** 0.5
        phase = torch.atan2(imag_input, real_input)
        phase[phase.ne(phase)] = 0.0
        return torch.stack((mag, phase), dim=input1.ndimension() - 1)
    else:
        """input1 is real part and input2 is imaginary part; both of size [B,C,H,W,D]"""
        mag = (input1 ** 2 + input2 ** 2) ** 0.5
        phase = torch.atan2(input2, input1)
        phase[phase.ne(phase)] = 0.0
        return mag, phase


class ZReLU(nn.Module):

    def __init__(self, polar=False):
        super(ZReLU, self).__init__()
        self.polar = polar

    def forward(self, input):
        ndims = input.ndimension()
        input_real = input.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        input_imag = input.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)
        if not self.polar:
            _mag, phase = cylindricalToPolarConversion(input_real, input_imag)
        else:
            phase = input_imag
        phase = phase.unsqueeze(-1)
        phase = torch.cat([phase, phase], ndims - 1)
        output = torch.where(phase >= 0.0, input, torch.tensor(0.0))
        output = torch.where(phase <= np.pi / 2, output, torch.tensor(0.0))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 2])]


def get_init_inputs():
    return [[], {}]
