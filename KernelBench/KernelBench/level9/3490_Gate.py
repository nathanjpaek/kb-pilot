import torch
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    """Gate Unit
	g = sigmoid(Wx)
	x = g * x
	"""

    def __init__(self, input_size):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        """
		Args:
			x: batch * len * dim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			res: batch * len * dim
		"""
        x_proj = self.linear(x)
        gate = F.sigmoid(x)
        return x_proj * gate


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
