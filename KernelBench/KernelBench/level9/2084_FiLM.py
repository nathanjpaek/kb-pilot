import torch
import torch.nn as nn


class FiLM(nn.Module):
    """ Feature-wise Linear Modulation (FiLM) layer"""

    def __init__(self, input_size, output_size, num_film_layers=1,
        layer_norm=False):
        """
        :param input_size: feature size of x_cond
        :param output_size: feature size of x_to_film
        :param layer_norm: true or false
        """
        super(FiLM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else None
        film_output_size = self.output_size * num_film_layers * 2
        self.gb_weights = nn.Linear(self.input_size, film_output_size)
        self.gb_weights.bias.data.fill_(0)

    def forward(self, x_cond, x_to_film):
        gb = self.gb_weights(x_cond).unsqueeze(1)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        out = (1 + gamma) * x_to_film + beta
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
