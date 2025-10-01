import torch
import torch.cuda
import torch
import torch.nn as nn
import torch.nn.functional as F


class outputCNN(nn.Module):

    def __init__(self, input_dim):
        super(outputCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=128,
            kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=(5, 5), padding=(2, 2))
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1,
            kernel_size=(5, 5), padding=(2, 2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        output_size = x.shape
        x, i = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.unpool(x, i, output_size=output_size)
        x = F.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
