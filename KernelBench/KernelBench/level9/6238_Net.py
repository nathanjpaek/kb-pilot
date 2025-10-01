import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_seq_length, output_num_classes):
        """Initialize model layers"""
        super(Net, self).__init__()
        self.input_seq_length = input_seq_length
        self.output_num_classes = output_num_classes
        self.fc1 = nn.Linear(self.input_seq_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_num_classes)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        """Forward pass through the model"""
        x = x.view(x.shape[0], self.input_seq_length)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_seq_length': 4, 'output_num_classes': 4}]
