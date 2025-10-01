from torch.autograd import Function
import math
import numbers
import torch
import numpy as np
import torch.nn as nn


def hsv2rgb(hsv):
    """Convert a 4-d HSV tensor to the RGB counterpart.
    >>> %timeit hsv2rgb_lookup(hsv)
    2.37 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit hsv2rgb(rgb)
    298 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> torch.allclose(hsv2rgb(hsv), hsv2rgb_lookup(hsv), atol=1e-6)
    True
    References
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    """
    h, s, v = hsv[:, [0]], hsv[:, [1]], hsv[:, [2]]
    c = v * s
    n = hsv.new_tensor([5, 3, 1]).view(3, 1, 1)
    k = (n + h * 6) % 6
    t = torch.min(k, 4.0 - k)
    t = torch.clamp(t, 0, 1)
    return v - c * t


def rgb2hsv(rgb):
    """Convert a 4-d RGB tensor to the HSV counterpart.
    Here, we compute hue using atan2() based on the definition in [1],
    instead of using the common lookup table approach as in [2, 3].
    Those values agree when the angle is a multiple of 30°,
    otherwise they may differ at most ~1.2°.
    >>> %timeit rgb2hsv_lookup(rgb)
    1.07 ms ± 2.96 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> %timeit rgb2hsv(rgb)
    380 µs ± 555 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> (rgb2hsv_lookup(rgb) - rgb2hsv(rgb)).abs().max()
    tensor(0.0031, device='cuda:0')
    References
    [1] https://en.wikipedia.org/wiki/Hue
    [2] https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [3] https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L212
    """
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    hue = torch.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)
    hue = hue % (2 * math.pi) / (2 * math.pi)
    saturate = 1 - Cmin / (Cmax + 1e-08)
    value = Cmax
    hsv = torch.stack([hue, saturate, value], dim=1)
    hsv[~torch.isfinite(hsv)] = 0.0
    return hsv


class RandomHSVFunction(Function):

    @staticmethod
    def forward(ctx, x, f_h, f_s, f_v):
        x = rgb2hsv(x)
        h = x[:, 0, :, :]
        h += f_h * 255.0 / 360.0
        h = h % 1
        x[:, 0, :, :] = h
        x[:, 1, :, :] = x[:, 1, :, :] * f_s
        x[:, 2, :, :] = x[:, 2, :, :] * f_v
        x = torch.clamp(x, 0, 1)
        x = hsv2rgb(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


class ColorJitterLayer(nn.Module):

    def __init__(self, brightness, contrast, saturation, hue):
        super(ColorJitterLayer, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5
            ), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')),
        clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    'If {} is a single number, it must be non negative.'.
                    format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError('{} values should be between {}'.format(
                    name, bound))
        else:
            raise TypeError(
                '{} should be a single number or a list/tuple with lenght 2.'
                .format(name))
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)
            means = torch.mean(x, dim=[2, 3], keepdim=True)
            x = (x - means) * factor + means
        return torch.clamp(x, 0, 1)

    def adjust_hsv(self, x):
        f_h = x.new_zeros(x.size(0), 1, 1)
        f_s = x.new_ones(x.size(0), 1, 1)
        f_v = x.new_ones(x.size(0), 1, 1)
        if self.hue:
            f_h.uniform_(*self.hue)
        if self.saturation:
            f_s = f_s.uniform_(*self.saturation)
        if self.brightness:
            f_v = f_v.uniform_(*self.brightness)
        return RandomHSVFunction.apply(x, f_h, f_s, f_v)

    def transform(self, inputs):
        if np.random.rand() > 0.5:
            transforms = [self.adjust_contrast, self.adjust_hsv]
        else:
            transforms = [self.adjust_hsv, self.adjust_contrast]
        for t in transforms:
            inputs = t(inputs)
        return inputs

    def forward(self, inputs):
        return self.transform(inputs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'brightness': 4, 'contrast': 4, 'saturation': 4, 'hue': 4}]
