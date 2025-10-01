import torch
import torch.nn as nn


class Upsample(nn.Upsample):
    """
		Upsampling via interporlation

		Args:
			x:	(N, T, C)
		Returns:
			y:	(N, S * T, C)
					(S: scale_factor)
	"""

    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = super(Upsample, self).forward(x)
        x = x.transpose(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
