from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class _CNN(nn.Module):

    def __init__(self, config):
        super(_CNN, self).__init__()
        self.config = config
        self.in_channels = 1
        self.in_height = self.config.max_length
        self.in_width = self.config.word_size + 2 * self.config.pos_size + 100
        self.kernel_size = self.config.window_size, self.in_width
        self.out_channels = self.config.hidden_size
        self.stride = 1, 1
        self.padding = 1, 0
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.
            kernel_size, self.stride, self.padding)

    def forward(self, embedding):
        return self.cnn(embedding)


def get_inputs():
    return [torch.rand([4, 1, 128, 128])]


def get_init_inputs():
    return [[], {'config': _mock_config(max_length=4, word_size=4, pos_size
        =4, window_size=4, hidden_size=4)}]
