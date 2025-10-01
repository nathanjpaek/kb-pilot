import torch
import torch.nn as nn
import torch.optim
import torch.utils.data


class Conv1d2Score(nn.Module):
    """Calculate a N*out_dim tensor from N*in_dim*seq_len using nn.Conv1d
  Essentially it is a linear layer
  
  Args:
    in_dim: int
    out_dim: int, usually number of classes
    seq_len: int
    
  Shape:
    - Input: N*in_dim*seq_len
    - Output: N*out_dim
    
  Attributes:
    weight (Tensor): the learnable weights of the module of shape 
      out_channels (out_dim) * in_channels (in_dim) * kernel_size (seq_len)
    bias (Tensor): shape: out_channels (out_dim)
    
  Examples::
  
    >>> x = torch.randn(2, 3, 4, device=device)
    >>> model = Conv1d2Score(3, 5, 4)
    >>> model(x).shape
  """

    def __init__(self, in_dim, out_dim, seq_len, bias=True):
        super(Conv1d2Score, self).__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=seq_len, bias=bias)

    def forward(self, x):
        out = self.conv(x).squeeze(-1)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'seq_len': 4}]
