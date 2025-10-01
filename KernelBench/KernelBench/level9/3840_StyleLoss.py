import torch
import torch.nn as nn


class StyleLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram(self, feature):
        N, C, H, W = feature.shape
        feature = feature.view(N, C, H * W)
        gram_mat = torch.bmm(feature, torch.transpose(feature, 1, 2))
        return gram_mat / (C * H * W)

    def forward(self, results, targets):
        loss = 0.0
        for i, (ress, tars) in enumerate(zip(results, targets)):
            loss += self.l1loss(self.gram(ress), self.gram(tars))
        return loss / len(results)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
