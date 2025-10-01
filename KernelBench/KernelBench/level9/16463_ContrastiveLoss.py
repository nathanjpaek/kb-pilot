import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn.parallel
import torch.optim


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True
            )
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2).
            squeeze() * label.float() + torch.pow(torch.clamp(self.margin -
            euclidean_distance, min=0.0), 2).squeeze() * (1 - label.float()))
        return loss_contrastive


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
