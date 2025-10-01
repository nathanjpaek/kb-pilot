import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, size, upscale_factor, layer_size, channels):
        super(DQN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=size ** 2, out_features=layer_size)
        self.fc2 = nn.Linear(in_features=layer_size, out_features=size ** 2)

    def forward(self, input_image):
        image_vector = input_image.view(input_image.size(0), -1)
        x = self.relu(self.fc1(image_vector))
        reconstruction = self.fc2(x)
        return reconstruction


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4, 'upscale_factor': 1.0, 'layer_size': 1,
        'channels': 4}]
