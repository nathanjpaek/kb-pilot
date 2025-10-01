import math
import torch
import numpy as np
import torch.optim
import torch.utils.data


class fChannelAttentionGG(torch.nn.Module):

    def __init__(self, N_h_in, N_in, ratio=1, group='SE2'):
        super(fChannelAttentionGG, self).__init__()
        self.N_in = N_in
        self.ratio = ratio
        self.N_h_in = N_h_in
        self.N_h = N_h_in
        self.weight_fc1 = torch.nn.Parameter(torch.rand(self.N_in // ratio,
            self.N_in, self.N_h_in))
        self.weight_fc2 = torch.nn.Parameter(torch.rand(self.N_in, self.
            N_in // ratio, self.N_h_in))
        self.action = self._left_action_of_h_grid_se2
        if group == 'E2':
            group = importlib.import_module('attgconv.group.' + group)
            e2_layers = attgconv.layers(group)
            n_grid = 8
            self.h_grid = e2_layers.H.grid_global(n_grid)
            self.action = self._left_action_on_grid_e2
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_fc1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight_fc2, a=math.sqrt(5))

    def forward(self, input):
        fc1, fc2 = self.action()
        input_mean = input.mean(dim=[-2, -1]).unsqueeze(-1)
        input_max = input.max(dim=-2)[0].max(dim=-1)[0].unsqueeze(-1)
        avg_out = self._linear(torch.relu(self._linear(input_mean, fc1)), fc2)
        max_out = self._linear(torch.relu(self._linear(input_max, fc1)), fc2)
        out = torch.sigmoid(avg_out + max_out)
        out = torch.reshape(out, [out.shape[0], self.N_in, self.N_h_in, 1, 1])
        return out

    def _linear(self, input, w):
        in_reshaped = input.unsqueeze(-4).unsqueeze(-5)
        w_reshaped = torch.reshape(w, [1, w.shape[0], w.shape[1], w.shape[2
            ], w.shape[3], 1])
        output = (in_reshaped * w_reshaped).sum(dim=[-3, -2])
        return output

    def _left_action_of_h_grid_se2(self):
        fc1 = torch.stack([self.weight_fc1.roll(shifts=i, dims=-1) for i in
            range(self.N_h)], dim=1)
        fc2 = torch.stack([self.weight_fc2.roll(shifts=i, dims=-1) for i in
            range(self.N_h)], dim=1)
        return fc1, fc2

    def _left_action_on_grid_e2(self):
        fc1 = torch.stack([self._left_action_of_h_grid_e2(h, self.
            weight_fc1) for h in self.h_grid.grid], dim=1)
        fc2 = torch.stack([self._left_action_of_h_grid_e2(h, self.
            weight_fc2) for h in self.h_grid.grid], dim=1)
        return fc1, fc2

    def _left_action_of_h_grid_e2(self, h, fx):
        shape = fx.shape
        Lgfx = fx.clone()
        Lgfx = torch.reshape(Lgfx, [shape[0], shape[1], 2, 4])
        if h[0] != 0:
            Lgfx[:, :, 0, :] = torch.roll(Lgfx[:, :, 0, :], shifts=int(
                torch.round(1.0 / (np.pi / 2.0) * h[0]).item()), dims=-1)
            Lgfx[:, :, 1, :] = torch.roll(Lgfx[:, :, 1, :], shifts=-int(
                torch.round(1.0 / (np.pi / 2.0) * h[0]).item()), dims=-1)
        if h[-1] == -1:
            Lgfx = torch.roll(Lgfx, shifts=1, dims=-2)
        Lgfx = torch.reshape(Lgfx, shape)
        return Lgfx


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'N_h_in': 4, 'N_in': 4}]
