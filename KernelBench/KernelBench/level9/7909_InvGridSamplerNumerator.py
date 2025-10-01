import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch.utils.data


def ravel_multi_index(indices, shape):
    indices_ravel = indices[0]
    for i in range(1, len(indices)):
        indices_ravel = indices_ravel * shape[i] + indices[i]
    return indices_ravel


def add_repeated(t, indices, values):
    shape = t.shape
    t_ravel = t.view(t.numel())
    indices_ravel = ravel_multi_index(indices, shape)
    t_ravel.index_add_(0, indices_ravel, values)


def meshgrid_tensor(*sizes, normalize=True, device='cuda:0'):
    if not normalize:
        aranges = [torch.arange(cur_size, device=device) for cur_size in sizes]
        grids = torch.meshgrid(aranges)
        grid = torch.stack(grids, dim=-1)
    else:
        aranges = [torch.arange(cur_size, device=device).float() for
            cur_size in sizes]
        grids = torch.meshgrid(aranges)
        grid = np.stack([(cur_grid / float(max(sizes[i] - 1, 1))) for i,
            cur_grid in enumerate(grids)], dim=-1)
    return grid


class InvGridSamplerNumerator(nn.Module):
    eps = 1e-10

    def __init__(self, OH=None, OW=None):
        super(InvGridSamplerNumerator, self).__init__()
        self.OH = OH
        self.OW = OW

    def forward(self, x, inv_grid):
        eps = InvGridSamplerNumerator.eps
        batch_size, n_channels, h, w = x.size(0), x.size(1), x.size(2
            ) if self.OH is None else self.OH, x.size(3
            ) if self.OW is None else self.OW
        inv_grid = (inv_grid.clone() + 1) / 2.0
        inv_grid[..., 0] *= h
        inv_grid[..., 1] *= w
        inv_grid += 1
        inv_grid = torch.stack([inv_grid[..., 0].clamp(0, h + 1 - 2 * eps),
            inv_grid[..., 1].clamp(0, w + 1 - 2 * eps)], dim=-1)
        inv_grid = inv_grid[:, np.newaxis].repeat(1, n_channels, 1, 1, 1)
        A = torch.zeros((batch_size, n_channels, h + 3, w + 3), device=x.device
            )
        mgrid = meshgrid_tensor(batch_size, n_channels, x.size(2), x.size(3
            ), normalize=False, device=inv_grid.device)
        input_cells = mgrid.view(-1, mgrid.size(4))
        input_inds_b, input_inds_ch, _input_inds_i, _input_inds_j = (
            input_cells[..., 0], input_cells[..., 1], input_cells[..., 2],
            input_cells[..., 3])
        output_inds_b = input_inds_b
        output_inds_ch = input_inds_ch
        output_cells_float = inv_grid.view(-1, inv_grid.size(4))
        output_cells_long = output_cells_float.long()
        for di in range(0, 2):
            output_inds_i = output_cells_long[..., 0] + di
            corners_i = output_inds_i.float()
            bilinear_weights_i = F.relu(1 - torch.abs(output_cells_float[
                ..., 0] - corners_i))
            for dj in range(0, 2):
                output_inds_j = output_cells_long[..., 1] + dj
                corners_j = output_inds_j.float()
                bilinear_weights = bilinear_weights_i * F.relu(1 - torch.
                    abs(output_cells_float[..., 1] - corners_j))
                add_repeated(A, (output_inds_b, output_inds_ch,
                    output_inds_i, output_inds_j), x.reshape(-1) *
                    bilinear_weights)
        A = A[..., 1:h + 1, 1:w + 1]
        return A


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
