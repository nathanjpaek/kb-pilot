import torch
import torch.nn as nn
import torch.fft


class CNN(nn.Module):
    """Convolutional Neural Networks."""

    def __init__(self, input_size, hidden_dim, output_size):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=input_size, out_channels=
            hidden_dim, kernel_size=3, padding=1, padding_mode='circular',
            bias=False)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim * output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.Conv1(x).transpose(1, 2)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_dim': 4, 'output_size': 4}]
