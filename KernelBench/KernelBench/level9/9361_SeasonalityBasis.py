import torch
import numpy as np
import torch as t


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """

    def __init__(self, harmonics: 'int', backcast_size: 'int',
        forecast_size: 'int'):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32), np.arange
            (harmonics, harmonics / 2 * forecast_size, dtype=np.float32) /
            harmonics)[None, :]
        backcast_grid = -2 * np.pi * (np.arange(backcast_size, dtype=np.
            float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (np.arange(forecast_size, dtype=np.
            float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(
            np.cos(backcast_grid)), dtype=t.float32), requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(
            np.sin(backcast_grid)), dtype=t.float32), requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(
            np.cos(forecast_grid)), dtype=t.float32), requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(
            np.sin(forecast_grid)), dtype=t.float32), requires_grad=False)

    def forward(self, theta: 't.Tensor'):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 *
            params_per_harmonic:3 * params_per_harmonic], self.
            backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 *
            params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, :
            params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:,
            params_per_harmonic:2 * params_per_harmonic], self.
            forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos
        return backcast, forecast


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'harmonics': 4, 'backcast_size': 4, 'forecast_size': 4}]
