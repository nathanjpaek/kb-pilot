import torch
import torch.nn as nn
import torch.nn.utils


class Highway(nn.Module):

    def __init__(self, e_word):
        """ Init Highway.

        @param e_word (int): Output embedding size of target word.
        """
        super(Highway, self).__init__()
        self.proj_layer = nn.Linear(e_word, e_word)
        self.gate_layer = nn.Linear(e_word, e_word)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out):
        """ Forward pass of Highway.

        @param x_conv_out (Tensor): tensor from convolutional layer, shape (batch_size, e_word)
        
        @returns x_highway (Tensor): output tensor after highway layer, shape (batch_size, e_word)
        """
        x_proj = self.ReLU(self.proj_layer(x_conv_out))
        x_gate = self.sigmoid(self.gate_layer(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'e_word': 4}]
