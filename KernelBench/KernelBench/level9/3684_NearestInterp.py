import torch


class NearestInterp(torch.nn.Module):
    """

  Nearest neighbor interpolation layer.

  note:
  From the source code, it appears that Darknet uses
  nearest neighbor method for its upsampling layer
  (darknet master-30 oct 2018).

  Internally calls torch.nn.functional.interpolate
  to suppress the warning on calling
  torch.nn.Upsample.

  """

    def __init__(self, factor):
        """
    Constructor.

    Parameters
    ----------
    factor: float
    The interpolation factor.

    """
        super(NearestInterp, self).__init__()
        self.factor = factor

    def forward(self, inp):
        """

    Parameters
    ----------
    inp: torch.tensor
    Input image as torch rank-4 tensor: batch x
    channels x height x width.

    """
        return torch.nn.functional.interpolate(inp, scale_factor=self.
            factor, mode='nearest')


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'factor': 4}]
