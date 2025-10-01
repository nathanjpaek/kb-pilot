import torch


class TensorRepeat(torch.nn.Module):
    """
  duolicate a 1D tensor into N channels (grayscale to rgb for instance)
  code derived from https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
  """

    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(self, tensor):
        return tensor.repeat(self.num_output_channels, 1, 1)

    def __repr__(self):
        return (self.__class__.__name__ +
            f'(num_output_channels={self.num_output_channels})')


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
