import torch
import torch.nn as nn
import torch.utils.data.distributed


class silog_loss(nn.Module):

    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * d.mean() ** 2
            ) * 10.0


def get_inputs():
    return [torch.ones([4], dtype=torch.int64), torch.ones([4], dtype=torch
        .int64), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'variance_focus': 4}]
