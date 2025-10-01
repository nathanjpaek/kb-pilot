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


class NormalizeScaleController(ScaleControllerBase):

    def __init__(self):
        super(NormalizeScaleController, self).__init__()

    def forward(self, x: 'torch.Tensor', dim: 'int'=-1, p: 'int'=1):
        return torch.nn.functional.normalize(x, p=p, dim=dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
