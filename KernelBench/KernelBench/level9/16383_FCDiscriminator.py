from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

    def __init__(self, options):
        """
        The fully connected generator is initialized by creating a chain of
        fully connected layers that perform transformations

            n * n -> 2^(k - 1) * d -> ... -> 2 * d -> 1

        where
            d = options.state_size
            k = options.discriminator_layers
            n = options.image_size
        """
        super(FCDiscriminator, self).__init__()
        self.dropout = options.discriminator_dropout
        self.layers = options.discriminator_layers
        sizes = [options.image_size * options.image_size]
        size = options.state_size * 2 ** (options.discriminator_layers - 1)
        for i in range(options.discriminator_layers - 1):
            sizes.append(size)
            size //= 2
        sizes.append(1)
        for i in range(options.discriminator_layers):
            layer = nn.Linear(sizes[i], sizes[i + 1])
            self.add_module(f'linear_{i}', layer)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
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
        return torch.sigmoid(x)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'options': _mock_config(discriminator_dropout=0.5,
        discriminator_layers=1, image_size=4, state_size=4)}]
