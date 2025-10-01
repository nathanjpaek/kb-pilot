from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class FuseLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig -
            input1, orig * input1], dim=-1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig -
            input2, orig * input2], dim=-1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim=-1)))
        return fuse_prob * input1 + (1 - fuse_prob) * input2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
