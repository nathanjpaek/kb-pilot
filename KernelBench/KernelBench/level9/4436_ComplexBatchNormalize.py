import torch
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


def polarToCylindricalConversion(input1, input2=None):
    if input2 is None:
        """input1 is tensor of [B,C,H,W,D,2] contains both magnitude and phase channels
         in the last dims"""
        ndims = input1.ndimension()
        mag_input = input1.narrow(ndims - 1, 0, 1).squeeze(ndims - 1)
        phase_input = input1.narrow(ndims - 1, 1, 1).squeeze(ndims - 1)
        real = mag_input * torch.cos(phase_input)
        imag = mag_input * torch.sin(phase_input)
        return torch.stack((real, imag), dim=input1.ndimension() - 1)
    else:
        """input1 is magnitude part and input2 is phase part; both of size [B,C,H,W,D]"""
        real = input1 * torch.cos(input2)
        imag = input1 * torch.sin(input2)
        return real, imag


def normalizeComplexBatch_byMagnitudeOnly(x, polar=False):
    """ normalize the complex batch by making the magnitude of mean 1 and std 1, and keep the phase as it is"""
    ndims = x.ndimension()
    shift_mean = 1
    if not polar:
        x = cylindricalToPolarConversion(x)
    if ndims == 4:
        mag = x[:, :, :, 0]
        mdims = mag.ndimension()
        mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2]))
        normalized_mag = (mag - torch.mean(mag_shaped, mdims - 1, keepdim=
            True).unsqueeze(mdims)) / torch.std(mag_shaped, mdims - 1,
            keepdim=True).unsqueeze(mdims) + shift_mean
        x = torch.stack([normalized_mag, x[:, :, :, :, 1]], dim=3)
    elif ndims == 5:
        mag = x[:, :, :, :, 0]
        mdims = mag.ndimension()
        mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2] *
            mag.shape[3]))
        normalized_mag = (mag - torch.mean(mag_shaped, mdims - 2, keepdim=
            True).unsqueeze(mdims - 1)) / torch.std(mag_shaped, mdims - 2,
            keepdim=True).unsqueeze(mdims - 1) + shift_mean
        x = torch.stack([normalized_mag, x[:, :, :, :, 1]], dim=4)
    elif ndims == 6:
        mag = x[:, :, :, :, :, 0]
        mdims = mag.ndimension()
        mag_shaped = mag.reshape((mag.shape[0], mag.shape[1], mag.shape[2] *
            mag.shape[3] * mag.shape[4]))
        normalized_mag = (mag - torch.mean(mag_shaped, mdims - 3, keepdim=
            True).unsqueeze(mdims - 2)) / torch.std(mag_shaped, mdims - 3,
            keepdim=True).unsqueeze(mdims - 2) + shift_mean
        x = torch.stack([normalized_mag, x[:, :, :, :, :, 1]], dim=5)
    x[x.ne(x)] = 0.0
    if not polar:
        x = polarToCylindricalConversion(x)
    return x


class ComplexBatchNormalize(nn.Module):

    def __init__(self):
        super(ComplexBatchNormalize, self).__init__()

    def forward(self, input):
        return normalizeComplexBatch_byMagnitudeOnly(input)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
