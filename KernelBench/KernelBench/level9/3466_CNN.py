import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils


class CNN(nn.Module):
    """
    Convolutional layer of a character-based convolutional encoder that outputs word embeddings.
    """

    def __init__(self, char_embed_size: 'int', word_embed_size: 'int',
        kernel_size: 'int'=5, padding: 'int'=1):
        """ Init CNN

        @param char_embed_size (int): size of the character embedding vector; in_channels (dimensionality)
        @param word_embed_size (int): size of the word embedding vector; out_channels (dimensionality)
        @param kernel_size (int): kernel size of the convolutional layer
        @param padding (int): padding size for convolutional layer
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(char_embed_size, word_embed_size, kernel_size
            =kernel_size, padding=padding)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """ Takes a minibatch of character embeddings of source sentences and computes convolutions in the temporal direction.
        Then, we take a max-pool over the temporal dimension to get the output.

        @param x (torch.Tensor): a minibatch of character-level word embeddings;
                                 shape (batch_size, char_embed_size, max_word_length)

        @returns x_conv_out (Tensor): a tensor of the result of convolution & max_pool; shape (batch_size, word_embed_size)
        """
        x_conv = F.relu(self.conv(x))
        num_windows = x_conv.shape[-1]
        x_conv_out = F.max_pool1d(x_conv, kernel_size=num_windows)
        x_conv_out = x_conv_out.squeeze(-1)
        return x_conv_out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'char_embed_size': 4, 'word_embed_size': 4}]
