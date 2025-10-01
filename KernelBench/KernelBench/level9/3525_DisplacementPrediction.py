import torch
import torch.nn as nn
import torch.utils.data


class DisplacementPrediction(nn.Module):

    def __init__(self, pedestrian_num, input_size, output_size):
        super(DisplacementPrediction, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, data):
        output_list = []
        for idx in range(0, self.pedestrian_num):
            output_list.append(self.fc1(data[:, idx]))
        output = torch.stack(output_list, 1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pedestrian_num': 4, 'input_size': 4, 'output_size': 4}]
