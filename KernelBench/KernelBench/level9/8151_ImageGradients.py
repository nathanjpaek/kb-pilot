import torch
import torch as th
import torch.utils.data


class ImageGradients(th.nn.Module):
    """
  Args:
    c_in(int): number of channels expected in the images.
    use_sobel(bool): if True, uses a (smoother) Sobel filter instead of simple
      finite differences.
  """

    def __init__(self, c_in, use_sobel=True):
        super(ImageGradients, self).__init__()
        if use_sobel:
            self.dx = th.nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=
                False, groups=c_in)
            self.dy = th.nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=
                False, groups=c_in)
            self.dx.weight.data.zero_()
            self.dx.weight.data[:, :, 0, 0] = -1
            self.dx.weight.data[:, :, 0, 2] = 1
            self.dx.weight.data[:, :, 1, 0] = -2
            self.dx.weight.data[:, :, 1, 2] = 2
            self.dx.weight.data[:, :, 2, 0] = -1
            self.dx.weight.data[:, :, 2, 2] = 1
            self.dy.weight.data.zero_()
            self.dy.weight.data[:, :, 0, 0] = -1
            self.dy.weight.data[:, :, 2, 0] = 1
            self.dy.weight.data[:, :, 0, 1] = -2
            self.dy.weight.data[:, :, 2, 1] = 2
            self.dy.weight.data[:, :, 0, 2] = -1
            self.dy.weight.data[:, :, 2, 2] = 1
        else:
            self.dx = th.nn.Conv2d(c_in, c_in, [1, 3], padding=(0, 1), bias
                =False, groups=c_in)
            self.dy = th.nn.Conv2d(c_in, c_in, [3, 1], padding=(1, 0), bias
                =False, groups=c_in)
            self.dx.weight.data.zero_()
            self.dx.weight.data[:, :, 0, 0] = -1
            self.dx.weight.data[:, :, 0, 1] = 1
            self.dy.weight.data.zero_()
            self.dy.weight.data[:, :, 0, 0] = -1
            self.dy.weight.data[:, :, 1, 0] = 1
        self.dx.weight.requires_grad = False
        self.dy.weight.requires_grad = False

    def forward(self, im):
        return th.cat([self.dx(im), self.dy(im)], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c_in': 4}]
