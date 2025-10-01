import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """Attention Layer merging Encoder and Decoder

    Attributes:
        hidden_size (int):
            The number of features in the hidden state h

    Reference:
        * https://github.com/bentrevett/pytorch-seq2seq
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: previous hidden state of the decoder [batch_size, hidden_size]
            encoder_outputs: outputs of encoder
                [batch_size, sample_size, num_directions*hidden_size]
        """
        sample_size = encoder_outputs.shape[1]
        hidden = hidden.squeeze(0)
        hidden = hidden.unsqueeze(1).repeat(1, sample_size, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs),
            dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
