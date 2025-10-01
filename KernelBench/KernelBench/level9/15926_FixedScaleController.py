import torch


class ScaleControllerBase(torch.nn.Module):
    """
    The base class for ScaleController.
    ScaleController is a callable class that re-scale input tensor's value.
    Traditional scale method may include:
        soft-max, L2 normalize, relu and so on.
    Advanced method:
        Learnable scale parameter
    """

    def __init__(self):
        super(ScaleControllerBase, self).__init__()

    def forward(self, x: 'torch.Tensor', dim: 'int'=0, p: 'int'=1):
        """
        Re-scale the input x into proper value scale.
        :param x: the input tensor
        :param dim: axis to scale(mostly used in traditional method)
        :param p: p parameter used in traditional methods
        :return: rescaled x
        """
        raise NotImplementedError


class FixedScaleController(ScaleControllerBase):
    """
    Scale parameter with a fixed value.
    """

    def __init__(self, normalizer: 'ScaleControllerBase'=None, scale_rate:
        'float'=50):
        super(FixedScaleController, self).__init__()
        self.scale_rate = scale_rate
        self.normalizer = normalizer

    def forward(self, x: 'torch.Tensor', dim: 'int'=0, p: 'int'=1):
        x = x * self.scale_rate
        if self.normalizer:
            x = self.normalizer(x, dim=dim, p=p)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
