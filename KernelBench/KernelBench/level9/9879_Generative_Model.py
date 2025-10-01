import torch
import torch.nn as nn


class Generative_Model(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2,
        output_size, n_classes):
        super(Generative_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.n_classes = n_classes
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.relu(output)
        output = self.fc3(output)
        output_features = output[:, 0:-self.n_classes]
        output_labels = output[:, -self.n_classes:]
        output_total = torch.cat((output_features, output_labels), 1)
        return output_total


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size_1': 4, 'hidden_size_2': 4,
        'output_size': 4, 'n_classes': 4}]
