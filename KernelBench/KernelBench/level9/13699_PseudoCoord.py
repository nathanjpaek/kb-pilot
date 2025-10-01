import torch
import torch.nn as nn
import torch.utils.data


class PseudoCoord(nn.Module):

    def __init__(self):
        super(PseudoCoord, self).__init__()

    def forward(self, b):
        """
        Input: 
        b: bounding box        [batch, num_obj, 4]  (x1,y1,x2,y2)
        Output:
        pseudo_coord           [batch, num_obj, num_obj, 2] (rho, theta)
        """
        batch_size, num_obj, _ = b.shape
        centers = (b[:, :, 2:] + b[:, :, :2]) * 0.5
        relative_coord = centers.view(batch_size, num_obj, 1, 2
            ) - centers.view(batch_size, 1, num_obj, 2)
        rho = torch.sqrt(relative_coord[:, :, :, 0] ** 2 + relative_coord[:,
            :, :, 1] ** 2)
        theta = torch.atan2(relative_coord[:, :, :, 0], relative_coord[:, :,
            :, 1])
        new_coord = torch.cat((rho.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)
        return new_coord


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
