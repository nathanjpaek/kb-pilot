import torch
import torch.utils.tensorboard
import torch.utils.data
import torch.distributed


class ConcatenateLinear(torch.nn.Module):
    """A torch module which concatenates several inputs and mixes them using a linear layer. """

    def __init__(self, left_size, right_size, output_size):
        """Creates a new concatenating linear layer.

        Parameters
        ----------
        left_size : int
            Size of the left input
        right_size : int
            Size of the right input
        output_size : int
            Size of the output.
        """
        super(ConcatenateLinear, self).__init__()
        self.left_size = left_size
        self.right_size = right_size
        self.output_size = output_size
        self._linear = torch.nn.Linear(left_size + right_size, output_size)

    def forward(self, left, right):
        return self._linear(torch.cat((left, right), dim=-1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'left_size': 4, 'right_size': 4, 'output_size': 4}]
