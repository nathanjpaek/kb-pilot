import torch
import torch.optim


class VisibilityFOV(torch.nn.Module):

    def __init__(self, width: 'int'=1, height: 'int'=1, coord_type: 'str'=
        'coord'):
        super(VisibilityFOV, self).__init__()
        self.coord_type = coord_type
        self.width = width
        self.height = height

    def forward(self, coords: 'torch.Tensor') ->torch.Tensor:
        _coords = coords.clone().detach()
        if self.coord_type != 'coord':
            _coords[..., 0] = (_coords[..., 0] + 1.0
                ) / 2.0 * self.width if self.coord_type == 'ndc' else _coords[
                ..., 0] * self.width
            _coords[..., 1] = (_coords[..., 1] + 1.0
                ) / 2.0 * self.height if self.coord_type == 'ndc' else _coords[
                ..., 1] * self.height
        masks = torch.zeros_like(coords)
        masks[..., 0] = (_coords[..., 0] >= 0) * (_coords[..., 0] < self.width)
        masks[..., 1] = (_coords[..., 1] >= 0) * (_coords[..., 1] < self.height
            )
        return masks


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
