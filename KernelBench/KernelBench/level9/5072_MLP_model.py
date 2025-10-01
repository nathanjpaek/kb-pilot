import torch
import torch.nn as nn


class MLP_model(nn.Module):

    def __init__(self, inputsize, layer1, layer2, layer3, device):
        super().__init__()
        self.fc1 = nn.Linear(inputsize, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, layer3)
        self.fc4 = nn.Linear(layer3, 1)
        self.device = device

    def forward(self, our_data):
        """
        our_data: [batch_size,1,4000]:[256,4000]
        output:[256,1]
        """
        mlp_output = nn.functional.relu(self.fc1(our_data))
        mlp_output = nn.functional.relu(self.fc2(mlp_output))
        mlp_output = nn.functional.relu(self.fc3(mlp_output))
        forecast_y = self.fc4(mlp_output)
        return forecast_y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputsize': 4, 'layer1': 1, 'layer2': 1, 'layer3': 1,
        'device': 0}]
