import torch
import torch.nn as nn
import torch.nn.utils


class CNN(nn.Module):

    def __init__(self, e_char, e_word):
        """ Init CNN.

        @param e_word (int): Output embedding size of target char.
        @param e_word (int): Output embedding size of target word.
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=e_word,
            kernel_size=5, padding=1)
        self.ReLU = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x_reshaped):
        """ Forward pass of CNN.

        @param x_reshaped (Tensor): tensor after padding and embedding lookup, shape (src_len * batch_size, e_char, m_word)
        
        @returns x_conv_out (Tensor): output tensor after highway layer, shape (src_len * batch_size, e_word)
        """
        relu = self.ReLU(self.conv(x_reshaped))
        x_conv_out = self.maxpool(relu).squeeze(-1)
        return x_conv_out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'e_char': 4, 'e_word': 4}]
