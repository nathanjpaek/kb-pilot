import torch
from torch import nn
from torch.nn import CosineSimilarity


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = CosineSimilarity()
        self.eps = 1e-06
        self.mse = torch.nn.MSELoss()

    def forward(self, output1, output2, target, size_average=True):
        distances = self.distance(output1, output2)
        losses = (1 - target.float()) * nn.functional.relu(self.margin -
            distances).pow(2) + target.float() * (1 - distances).pow(2) / 4
        return losses.mean() if size_average else losses.sum(), distances


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
