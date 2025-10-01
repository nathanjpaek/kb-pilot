import torch
import torch.nn as nn


class DeepNeuralNetwork(nn.Module):

    def __init__(self, u):
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, u)
        self.fc2 = nn.Linear(u, u)
        self.fc3 = nn.Linear(u, u)
        self.fc4 = nn.Linear(u, 1)
        self.ReLu = nn.ReLU()
        self.Sigmoid = nn.Softsign()
        self.Tanhshrink = nn.Tanh()
        self.Softplus = nn.ELU()

    def forward(self, x):
        layer1 = x.view(-1, 1)
        layer2 = self.ReLu(self.fc1(layer1))
        layer3 = self.Sigmoid(self.fc2(layer2))
        layer4 = self.Tanhshrink(self.fc3(layer3))
        layer5 = self.Tanhshrink(self.fc4(layer4))
        return layer5

    def __repr__(self):
        return json.dumps(self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'u': 4}]
