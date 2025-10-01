import torch
import torch.nn as nn
import torch.nn.functional as F


class Liner_Qnet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.L1 = nn.Linear(input_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        return x

    def save(self, fname='model.pth'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_path = os.path.join(model_path, fname)
        torch.save(self, model_path)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
