from _paritybench_helpers import _mock_config
import torch
import numpy as np


class PopulationColourRGBTransforms(torch.nn.Module):
    """RGB color transforms and ordering of patches."""

    def __init__(self, config, device, num_patches=1, pop_size=1,
        requires_grad=True):
        super(PopulationColourRGBTransforms, self).__init__()
        self.config = config
        self.device = device
        None
        self._pop_size = pop_size
        None
        rgb_init_range = self.config['initial_max_rgb'] - self.config[
            'initial_min_rgb']
        population_reds = np.random.rand(pop_size, num_patches, 1, 1, 1
            ) * rgb_init_range + self.config['initial_min_rgb']
        population_greens = np.random.rand(pop_size, num_patches, 1, 1, 1
            ) * rgb_init_range + self.config['initial_min_rgb']
        population_blues = np.random.rand(pop_size, num_patches, 1, 1, 1
            ) * rgb_init_range + self.config['initial_min_rgb']
        population_zeros = np.ones((pop_size, num_patches, 1, 1, 1))
        population_orders = np.random.rand(pop_size, num_patches, 1, 1, 1)
        self.reds = torch.nn.Parameter(torch.tensor(population_reds, dtype=
            torch.float), requires_grad=requires_grad)
        self.greens = torch.nn.Parameter(torch.tensor(population_greens,
            dtype=torch.float), requires_grad=requires_grad)
        self.blues = torch.nn.Parameter(torch.tensor(population_blues,
            dtype=torch.float), requires_grad=requires_grad)
        self._zeros = torch.nn.Parameter(torch.tensor(population_zeros,
            dtype=torch.float), requires_grad=False)
        self.orders = torch.nn.Parameter(torch.tensor(population_orders,
            dtype=torch.float), requires_grad=requires_grad)

    def _clamp(self):
        self.reds.data = self.reds.data.clamp(min=self.config['min_rgb'],
            max=self.config['max_rgb'])
        self.greens.data = self.greens.data.clamp(min=self.config['min_rgb'
            ], max=self.config['max_rgb'])
        self.blues.data = self.blues.data.clamp(min=self.config['min_rgb'],
            max=self.config['max_rgb'])
        self.orders.data = self.orders.data.clamp(min=0.0, max=1.0)

    def copy_and_mutate_s(self, parent, child):
        with torch.no_grad():
            self.reds[child, ...] = self.reds[parent, ...] + self.config[
                'colour_mutation_scale'] * torch.randn(self.reds[child, ...
                ].shape)
            self.greens[child, ...] = self.greens[parent, ...] + self.config[
                'colour_mutation_scale'] * torch.randn(self.greens[child,
                ...].shape)
            self.blues[child, ...] = self.blues[parent, ...] + self.config[
                'colour_mutation_scale'] * torch.randn(self.blues[child,
                ...].shape)
            self.orders[child, ...] = self.orders[parent, ...]

    def copy_from(self, other, idx_to, idx_from):
        """Copy parameters from other colour transform, for selected indices."""
        assert idx_to < self._pop_size
        with torch.no_grad():
            self.reds[idx_to, ...] = other.reds[idx_from, ...]
            self.greens[idx_to, ...] = other.greens[idx_from, ...]
            self.blues[idx_to, ...] = other.blues[idx_from, ...]
            self.orders[idx_to, ...] = other.orders[idx_from, ...]

    def forward(self, x):
        self._clamp()
        colours = torch.cat([self.reds, self.greens, self.blues, self.
            _zeros, self.orders], 2)
        return colours * x

    def tensor_to(self, device):
        self.reds = self.reds
        self.greens = self.greens
        self.blues = self.blues
        self.orders = self.orders
        self._zeros = self._zeros


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(initial_max_rgb=4, initial_min_rgb=
        4, min_rgb=4, max_rgb=4), 'device': 0}]
