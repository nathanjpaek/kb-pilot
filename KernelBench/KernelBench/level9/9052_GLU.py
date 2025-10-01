import torch
from torch import nn


class GLU(nn.Module):
    """
      The Gated Linear Unit GLU(a,b) = mult(a,sigmoid(b)) is common in NLP 
      architectures like the Gated CNN. Here sigmoid(b) corresponds to a gate 
      that controls what information from a is passed to the following layer. 

      Args:
          input_size (int): number defining input and output size of the gate
    """

    def __init__(self, input_size):
        super().__init__()
        self.a = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.b = nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): tensor passing through the gate
        """
        gate = self.sigmoid(self.b(x))
        x = self.a(x)
        return torch.mul(gate, x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
