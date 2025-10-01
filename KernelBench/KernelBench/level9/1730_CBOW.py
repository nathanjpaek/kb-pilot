import torch
import torch.nn as nn


class CBOW(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = sum([*x]).float()
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def save_model(self):
        torch.save(self.state_dict(), 'checkpoints/cbow_model.pt')

    def load_model(self):
        self.load_state_dict(torch.load('checkpoints/cbow_model.pt',
            map_location='cpu'))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
