import torch
import torch.utils.data


class FinalPool(torch.nn.Module):

    def __init__(self):
        super(FinalPool, self).__init__()

    def forward(self, input):
        """
		input : Tensor of shape (batch size, T, Cin)
		
		Outputs a Tensor of shape (batch size, Cin).
		"""
        return input.max(dim=1)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
