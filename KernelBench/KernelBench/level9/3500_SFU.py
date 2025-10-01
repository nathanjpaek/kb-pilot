import torch
import torch.nn as nn
import torch.nn.functional as F


class SFU(nn.Module):
    """Semantic Fusion Unit
	The ouput vector is expected to not only retrieve correlative information from fusion vectors,
	but also retain partly unchange as the input vector
	"""

    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = F.tanh(self.linear_r(r_f))
        g = F.sigmoid(self.linear_g(r_f))
        o = g * r + (1 - g) * x
        return o


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'fusion_size': 4}]
