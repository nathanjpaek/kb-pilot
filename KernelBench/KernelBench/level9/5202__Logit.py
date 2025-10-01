import torch


class _Logit(torch.nn.Module):
    """ Simple logistic regression model.
  """

    def __init__(self, din, dout=1):
        """ Model parameter constructor.
    Args:
      din  Number of input dimensions
      dout Number of output dimensions
    """
        super().__init__()
        self._din = din
        self._dout = dout
        self._linear = torch.nn.Linear(din, dout)

    def forward(self, x):
        """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
        return torch.sigmoid(self._linear(x.view(-1, self._din)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'din': 4}]
