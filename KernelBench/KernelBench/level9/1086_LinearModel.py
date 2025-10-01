import torch


class LinearModel(torch.nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int', dropout: 'float'
        ):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        data = data[:, 0, :]
        hidden = self.dropout(data)
        output = self.linear(hidden)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'dropout': 0.5}]
