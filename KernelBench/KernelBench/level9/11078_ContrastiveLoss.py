import torch
import torch.cuda
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Triplet loss function based on Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, output3, label):
        euclidean_distance = torch.mean(F.pairwise_distance(output1,
            output2, keepdim=True))
        euclidean_distance1 = torch.mean(F.pairwise_distance(output1,
            output3, keepdim=True))
        loss_contrastive = torch.mean((1 - label) * torch.pow(torch.clamp(
            self.margin + euclidean_distance1 - euclidean_distance, min=0.0
            ), 2) + label * torch.pow(torch.clamp(self.margin +
            euclidean_distance - euclidean_distance1, min=0.0), 2))
        return loss_contrastive


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
