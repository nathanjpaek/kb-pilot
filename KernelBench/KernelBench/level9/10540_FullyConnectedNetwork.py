import torch
import torch.nn.functional as F


class FullyConnectedNetwork(torch.nn.Module):

    def __init__(self, input_size, h1, h2, output_size):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_size, h1)
        self.layer_2 = torch.nn.Linear(h1, h2)
        self.layer_3 = torch.nn.Linear(h2, output_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'h1': 4, 'h2': 4, 'output_size': 4}]
