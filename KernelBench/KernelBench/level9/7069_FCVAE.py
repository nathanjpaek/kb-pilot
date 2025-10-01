import torch
from torch.nn import functional as F
from torch import nn


class BaseVAE(nn.Module):
    """
    Base abstract class for the Variational Autoencoders
    """

    def __init__(self, channels=1, width=28, height=28, z_dim=2):
        """
        Constructor

        Parameters:
            channels - The number of channels for the image
            width - The width of the image in pixels
            height - The height of the image in pixels
            z_dim - The dimension of the latent space
        """
        super(BaseVAE, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.z_dim = z_dim

    def getNbChannels(self):
        """
        Returns the number of channels of the handled images
        """
        return self.channels

    def getWidth(self):
        """
        Returns the width of the handled images in pixels
        """
        return self.width

    def getHeight(self):
        """
        Returns the height of the handled images in pixels
        """
        return self.height

    def getZDim(self):
        """
        Returns the dimension of the latent space of the VAE
        """
        return self.z_dim

    def flatten(self, x):
        """
        Can be used to flatten the output image. This method will only handle
        images of the original size specified for the network
        """
        return x.view(-1, self.channels * self.height * self.width)

    def unflatten(self, x):
        """
        Can be used to unflatten an image handled by the network. This method
        will only handle images of the original size specified for the network
        """
        return x.view(-1, self.channels, self.height, self.width)


class FCVAE(BaseVAE):
    """
    Fully connected Variational Autoencoder
    """

    def __init__(self, channels=1, width=28, height=28, z_dim=2):
        super(FCVAE, self).__init__(channels, width, height, z_dim)
        self.fc1 = nn.Linear(self.channels * self.width * self.height, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc4 = nn.Linear(400, self.channels * self.width * self.height)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(self.flatten(x))
        z = self.reparameterize(mu, logvar)
        return self.unflatten(self.decode(z)), mu, logvar


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
