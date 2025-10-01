import torch
import torch.nn as nn


class FeatExemplarAvgBlock(nn.Module):

    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        labels_train_transposed = labels_train.transpose(1, 2)
        weight_novel = torch.bmm(labels_train_transposed, features_train)
        weight_novel = weight_novel.div(labels_train_transposed.sum(dim=2,
            keepdim=True).expand_as(weight_novel))
        return weight_novel


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'nFeat': 4}]
