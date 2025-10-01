import torch
import torch.distributed
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed


class MSELoss(torch.nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, preds, heatmap_gt, weight):
        losses = 0.5 * weight * ((preds - heatmap_gt) ** 2).mean(dim=3).mean(
            dim=2)
        back_loss = losses.mean(dim=1).mean(dim=0)
        return back_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
