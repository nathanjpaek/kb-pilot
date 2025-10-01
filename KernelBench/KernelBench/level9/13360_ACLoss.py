import torch
import torch.utils.data


class ACLoss(torch.nn.Module):
    """Active Contour loss
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf

    Supports 2D and 3D data, as long as all spatial dimensions have the same
    size and there are only two output channels.

    Modifications:
    - Using mean instead of sum for reductions to avoid size dependency.
    - Instead of the proposed λ loss component weighting (which leads to
      exploding loss magnitudes for high λ values), a relative weight
      ``region_weight`` is used to balance the components:
      ``ACLoss = (1 - region_weight) * length_term + region_weight * region_term``
    """

    def __init__(self, num_classes: 'int', region_weight: 'float'=0.5):
        assert 0.0 <= region_weight <= 1.0, 'region_weight must be between 0 and 1'
        self.num_classes = num_classes
        self.region_weight = region_weight
        super().__init__()

    @staticmethod
    def get_length(output):
        if output.ndim == 4:
            dy = output[:, :, 1:, :] - output[:, :, :-1, :]
            dx = output[:, :, :, 1:] - output[:, :, :, :-1]
            dy = dy[:, :, 1:, :-2] ** 2
            dx = dx[:, :, :-2, 1:] ** 2
            delta_pred = torch.abs(dy + dx)
        elif output.ndim == 5:
            assert output.shape[3] == output.shape[4
                ], 'All spatial dims must have the same size'
            dz = output[:, :, 1:, :, :] - output[:, :, :-1, :, :]
            dy = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
            dx = output[:, :, :, :, 1:] - output[:, :, :, :, :-1]
            dz = dz[:, :, 1:, :-2, :-2] ** 2
            dy = dy[:, :, :-2, 1:, :-2] ** 2
            dx = dx[:, :, :-2, :-2, 1:] ** 2
            delta_pred = torch.abs(dz + dy + dx)
        length = torch.mean(torch.sqrt(delta_pred + 1e-06))
        return length

    @staticmethod
    def get_region(output, target):
        region_in = torch.mean(output * (target - 1.0) ** 2.0)
        region_out = torch.mean((1 - output) * target ** 2.0)
        return region_in + region_out

    def forward(self, output, target):
        assert output.shape[2] == output.shape[3
            ], 'All spatial dims must have the same size'
        if target.ndim == output.ndim - 1:
            target = torch.nn.functional.one_hot(target, self.num_classes
                ).transpose(1, -1)
        length_term = self.get_length(output
            ) if self.region_weight < 1.0 else 0.0
        region_term = self.get_region(output, target
            ) if self.region_weight > 0.0 else 0.0
        loss = (1 - self.region_weight
            ) * length_term + self.region_weight * region_term
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_classes': 4}]
