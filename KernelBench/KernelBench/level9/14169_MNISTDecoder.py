import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTDecoder(nn.Module):
    """
    MNIST decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of a fully
    connected layer of 128 units with ReLU activation followed by a convolutional block. The convolutional block
    consists fo 4 convolutional layers having 8, 8, 8  and 1 channels and a kernel size of 3. Each convolutional layer,
    except the last one, has ReLU nonlinearities and is followed by an upsampling layer of size 2. The final layers
    uses a sigmoid activation to clip the output values in [0, 1].
    """

    def __init__(self, latent_dim: 'int'):
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.conv1 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(3, 3))
        self.up3 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = F.relu(self.fc1(x))
        x = x.view(x.shape[0], 8, 4, 4)
        x = self.up1(F.relu(self.conv1(x)))
        x = self.up2(F.relu(self.conv2(x)))
        x = self.up3(F.relu(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4}]
