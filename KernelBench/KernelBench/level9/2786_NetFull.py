import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class NetFull(nn.Module):

    def __init__(self):
        super(NetFull, self).__init__()
        self.liner1 = nn.Linear(28 * 28, 400)
        self.liner2 = nn.Linear(400, 200)
        self.liner3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        input_output = F.tanh(self.liner1(x))
        hidden_1_output = F.tanh(self.liner2(input_output))
        hidden_2_output = self.liner3(hidden_1_output)
        output = F.log_softmax(input=hidden_2_output, dim=1)
        return output


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
