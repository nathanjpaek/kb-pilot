import torch


class PSN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PSN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc = torch.nn.Linear(self.input_size, self.hidden_size)

    def forward(self, x):
        x = self.fc(x)
        x = torch.prod(x, 1)
        x = torch.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
