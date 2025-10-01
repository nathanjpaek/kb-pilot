import torch
import torch.nn as nn


class LSGANLossGenerator(nn.Module):
    """
    This class implements the least squares generator GAN loss proposed in:
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf
    """

    def __init__(self) ->None:
        """
        Constructor method.
        """
        super(LSGANLossGenerator, self).__init__()

    def forward(self, discriminator_prediction_fake: 'torch.Tensor', **kwargs
        ) ->torch.Tensor:
        """
        Forward pass.
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Generator LSGAN loss
        """
        return -0.5 * (discriminator_prediction_fake - 1.0).pow(2).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
