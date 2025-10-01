import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(1e-05)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=1)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=1)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
