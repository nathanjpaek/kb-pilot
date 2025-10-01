import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data


class L2Conv2D(nn.Module):
    """
    Convolutional layer that computes the squared L2 distance instead of the conventional inner product. 
    """

    def __init__(self, num_prototypes, num_features, w_1, h_1):
        """
        Create a new L2Conv2D layer
        :param num_prototypes: The number of prototypes in the layer
        :param num_features: The number of channels in the input features
        :param w_1: Width of the prototypes
        :param h_1: Height of the prototypes
        """
        super().__init__()
        prototype_shape = num_prototypes, num_features, w_1, h_1
        self.prototype_vectors = nn.Parameter(torch.randn(prototype_shape),
            requires_grad=True)

    def forward(self, xs):
        """
        Perform convolution over the input using the squared L2 distance for all prototypes in the layer
        :param xs: A batch of input images obtained as output from some convolutional neural network F. Following the
                   notation from the paper, let the shape of xs be (batch_size, D, W, H), where
                     - D is the number of output channels of the conv net F
                     - W is the width of the convolutional output of F
                     - H is the height of the convolutional output of F
        :return: a tensor of shape (batch_size, num_prototypes, W, H) obtained from computing the squared L2 distances
                 for patches of the input using all prototypes
        """
        ones = torch.ones_like(self.prototype_vectors, device=xs.device)
        xs_squared_l2 = F.conv2d(xs ** 2, weight=ones)
        ps_squared_l2 = torch.sum(self.prototype_vectors ** 2, dim=(1, 2, 3))
        ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)
        xs_conv = F.conv2d(xs, weight=self.prototype_vectors)
        distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
        distance = torch.sqrt(torch.abs(distance) + 1e-14)
        if torch.isnan(distance).any():
            raise Exception(
                'Error: NaN values! Using the --log_probabilities flag might fix this issue'
                )
        return distance


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_prototypes': 4, 'num_features': 4, 'w_1': 4, 'h_1': 4}]
