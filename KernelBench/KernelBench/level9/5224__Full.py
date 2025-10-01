import torch


class _Full(torch.nn.Module):
    """ Simple, small fully connected model.
  """

    def __init__(self):
        """ Model parameter constructor.
    """
        super().__init__()
        self._f1 = torch.nn.Linear(28 * 28, 100)
        self._f2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
        x = torch.nn.functional.relu(self._f1(x.view(-1, 28 * 28)))
        x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
