import torch
import numpy as np
from torch import nn
from torch.nn.functional import relu


def regula_falsi(func, a, b, iterations):
    f_a = func(a, -1)
    f_b = func(b, -1)
    if torch.any(f_a * f_b >= 0):
        None
        raise Exception(
            'You have not assumed right initial values in regula falsi')
    c = a
    break_indices = torch.zeros_like(a).bool()
    for i in range(iterations):
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = func(c, i)
        break_indices[f_c == 0] = True
        b_eq_c_indices = (f_c * f_a < 0) & ~break_indices
        b[b_eq_c_indices] = c[b_eq_c_indices]
        a_eq_c_indices = ~(b_eq_c_indices | break_indices)
        a[a_eq_c_indices] = c[a_eq_c_indices]
    return c


class relu_constant_fraction(nn.Module):

    def __init__(self, nb_channels):
        super(relu_constant_fraction, self).__init__()
        self.biases = nn.Parameter(torch.zeros(nb_channels))
        self.biases.requires_grad = False
        self.bias_buffer = None

    def forward(self, x):
        return relu(x - self.biases.view(1, -1, 1, 1))

    def adjust_bias(self, desired_fraction, prev_layer_outputs):
        if desired_fraction > 1 - 0.001:
            self.biases.data = -10 * torch.ones_like(self.biases)
            return

        def get_fraction_deviation(biases, j):
            activations = relu(prev_layer_outputs - biases.view(1, -1, 1, 1))
            ratios = (activations > 0.001).float().mean(dim=(0, 2, 3))
            return ratios - desired_fraction
        with torch.no_grad():
            solutions = regula_falsi(get_fraction_deviation, -3 * torch.
                ones_like(self.biases), 3 * torch.ones_like(self.biases), 20)
            momentum = 0.75
            dampening = 0.0
            lr = 0.5
            delta = solutions - self.biases
            buf = self.bias_buffer
            if buf is None:
                buf = torch.clone(delta).detach()
                self.bias_buffer = buf
            else:
                buf.mul_(momentum).add_(delta, alpha=1 - dampening)
            delta = buf
            self.biases.add_(delta, alpha=lr)

    def get_activation_fractions(self, prev_layer_outputs):
        activations = relu(prev_layer_outputs - self.biases.view(1, -1, 1, 1))
        ratios = (activations > 0.001).float().mean(dim=(0, 2, 3))
        return ratios

    def show_trajectory(self, prev_layer_outputs):
        import matplotlib.pyplot as plt
        bias_values = np.linspace(-10, 10, 1000)
        fractions = np.zeros((1000, self.biases.shape[0]))
        for j, bias in enumerate(bias_values):
            cumulative_ratios = torch.zeros_like(self.biases)
            batch_size = 1000
            for i in range(0, len(prev_layer_outputs), batch_size):
                data = prev_layer_outputs[i:i + batch_size]
                activations = relu(data - bias)
                cumulative_ratios += (activations > 0.001).float().mean(dim
                    =(0, 2, 3)) * len(data)
            fractions[j] = (cumulative_ratios / len(prev_layer_outputs)
                ).detach().cpu().numpy()
        plt.plot(bias_values, fractions)
        plt.show()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nb_channels': 4}]
