import torch


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension,
            out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=200)
        self.output_layer = torch.nn.Linear(in_features=200, out_features=
            output_dimension)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        layer_1_output = self.relu(self.layer_1(input))
        layer_2_output = self.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dimension': 4, 'output_dimension': 4}]
