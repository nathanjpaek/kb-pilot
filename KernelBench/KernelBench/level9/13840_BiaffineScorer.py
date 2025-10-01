import torch
import torch.nn as nn


class BiaffineScorer(nn.Module):

    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1,
            output_size)
        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)
            ], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)
            ], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input1_size': 4, 'input2_size': 4, 'output_size': 4}]
