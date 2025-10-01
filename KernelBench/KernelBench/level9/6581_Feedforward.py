import torch


class Feedforward(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.fc1(x)
        y = self.sigmoid(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.fc3(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}]
