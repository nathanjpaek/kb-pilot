import torch
import torch.nn as nn


class ConvFunc(nn.Module):
    """Convolutional block, non-ODE.

    Parameters
    ----------
    device : torch.device

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    non_linearity : string
        One of 'relu' and 'softplus'
    """

    def __init__(self, device, img_size, num_filters, augment_dim=0,
        non_linearity='relu'):
        super(ConvFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.channels, self.height, self.width = img_size
        self.channels += augment_dim
        self.num_filters = num_filters
        self.nfe = 0
        self.conv1 = nn.Conv2d(self.channels, self.num_filters, kernel_size
            =1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
            kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_filters, self.channels, kernel_size
            =1, stride=1, padding=0)
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        out = self.conv1(x)
        out = self.non_linearity(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        out = self.conv3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0, 'img_size': [4, 4, 4], 'num_filters': 4}]
