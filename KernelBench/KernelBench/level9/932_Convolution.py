import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.parameter import Parameter
import torch.nn


def to_pair(data):
    """Converts a single or a tuple of data into a pair. If the data is a tuple with more than two elements, it selects
	the first two of them. In case of single data, it duplicates that data into a pair.

	Args:
		data (object or tuple): The input data.

	Returns:
		Tuple: A pair of data.
	"""
    if isinstance(data, tuple):
        return data[0:2]
    return data, data


class Convolution(nn.Module):
    """Performs a 2D convolution over an input spike-wave composed of several input
	planes. Current version only supports stride of 1 with no padding.

	The input is a 4D tensor with the size :math:`(T, C_{{in}}, H_{{in}}, W_{{in}})` and the crresponsing output
	is of size :math:`(T, C_{{out}}, H_{{out}}, W_{{out}})`, 
	where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
	:math:`H`, and :math:`W` are the hight and width of the input/output planes.

	* :attr:`in_channels` controls the number of input planes (channels/feature maps).

	* :attr:`out_channels` controls the number of feature maps in the current layer.

	* :attr:`kernel_size` controls the size of the convolution kernel. It can be a single integer or a tuple of two integers.

	* :attr:`weight_mean` controls the mean of the normal distribution used for initial random weights.

	* :attr:`weight_std` controls the standard deviation of the normal distribution used for initial random weights.

	.. note::

		Since this version of convolution does not support padding, it is the user responsibility to add proper padding
		on the input before applying convolution.

	Args:
		in_channels (int): Number of channels in the input.
		out_channels (int): Number of channels produced by the convolution.
		kernel_size (int or tuple): Size of the convolving kernel.
		weight_mean (float, optional): Mean of the initial random weights. Default: 0.8
		weight_std (float, optional): Standard deviation of the initial random weights. Default: 0.02
	"""

    def __init__(self, in_channels, out_channels, kernel_size, weight_mean=
        0.8, weight_std=0.02):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size)
        self.stride = 1
        self.bias = None
        self.dilation = 1
        self.groups = 1
        self.padding = 0
        self.weight = Parameter(torch.Tensor(self.out_channels, self.
            in_channels, *self.kernel_size))
        self.weight.requires_grad_(False)
        self.reset_weight(weight_mean, weight_std)

    def reset_weight(self, weight_mean=0.8, weight_std=0.02):
        """Resets weights to random values based on a normal distribution.

		Args:
			weight_mean (float, optional): Mean of the random weights. Default: 0.8
			weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
		"""
        self.weight.normal_(weight_mean, weight_std)

    def load_weight(self, target):
        """Loads weights with the target tensor.

		Args:
			target (Tensor=): The target tensor.
		"""
        self.weight.copy_(target)

    def forward(self, input):
        return fn.conv2d(input, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
