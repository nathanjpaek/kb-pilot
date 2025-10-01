from _paritybench_helpers import _mock_config
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


class ConditionalBottleNeck(nn.Module):
    """Down projection and up projection with FiLM layers within Transformer layer."""

    def __init__(self, config):
        super(ConditionalBottleNeck, self).__init__()
        self.emb_transf = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_modulation = FiLM(config.hidden_size, config.hidden_size)
        self.down_proj_layer = nn.Linear(config.hidden_size, config.
            hidden_size // 3)
        self.up_proj_layer = nn.Linear(config.hidden_size // 3, config.
            hidden_size)

    def forward(self, x_cond, hidden_states):
        x_cond = self.emb_transf(x_cond)
        hidden_states = self.hidden_modulation(x_cond=x_cond, x_to_film=
            hidden_states)
        hidden_states = self.down_proj_layer(hidden_states)
        hidden_states = self.up_proj_layer(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
