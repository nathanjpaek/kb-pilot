import torch
import torch.nn as nn


class Bilinear(nn.Module):

    def __init__(self, size):
        super(Bilinear, self).__init__()
        self.size = size
        self.mat = nn.Parameter(torch.FloatTensor(self.size, self.size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, vector1, vector2):
        bma = torch.matmul(vector1, self.mat).unsqueeze(1)
        ba = torch.matmul(bma, vector2.unsqueeze(2)).view(-1, 1)
        return ba


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
