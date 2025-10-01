import torch
from torch import nn
import torch.nn.functional as F


class Joiner(nn.Module):

    def __init__(self, input_dim: 'int', output_dim: 'int'):
        super().__init__()
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(self, encoder_out: 'torch.Tensor', decoder_out: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, C).
          decoder_out:
            Output from the decoder. Its shape is (N, U, C).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        assert encoder_out.ndim == decoder_out.ndim == 3
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == decoder_out.size(2)
        encoder_out = encoder_out.unsqueeze(2)
        decoder_out = decoder_out.unsqueeze(1)
        logit = encoder_out + decoder_out
        logit = F.relu(logit)
        output = self.output_linear(logit)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
