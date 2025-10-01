from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCGenerator(nn.Module):

    def __init__(self, options):
        """
        The fully connected generator is initialized by creating a chain of
        fully connected layers that perform transformations

            d -> 2 * d -> ... -> 2^(k - 1) * d -> n * n

        where
            d = options.state_size
            k = options.generator_layers
            n = options.image_size
        """
        super(FCGenerator, self).__init__()
        self.dropout = options.generator_dropout
        self.layers = options.generator_layers
        sizes = []
        size = options.state_size
        for i in range(options.generator_layers):
            sizes.append(size)
            size *= 2
        sizes.append(options.image_size * options.image_size)
        for i in range(options.generator_layers):
            layer = nn.Linear(sizes[i], sizes[i + 1])
            self.add_module(f'linear_{i}', layer)

    def forward(self, x):
        layers = {}
        for name, module in self.named_children():
            layers[name] = module
        for i in range(self.layers):
            layer = layers[f'linear_{i}']
            x = layer(x)
            if i < self.layers - 1:
                x = F.leaky_relu(x, 0.2)
                if self.dropout is not None:
                    x = F.dropout(x, self.dropout)
        return torch.tanh(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'options': _mock_config(generator_dropout=0.5,
        generator_layers=1, state_size=4, image_size=4)}]
