import math
import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class MutualInformationDiscriminatorHomo(nn.Module):

    def __init__(self, n_hidden, average_across_node_types=True,
        convex_combination_weight=None):
        super(MutualInformationDiscriminatorHomo, self).__init__()
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()
        self.average_across_node_types = average_across_node_types
        self.convex_combination_weight = convex_combination_weight
        self.global_summary = None

    def forward(self, positives, negatives):
        l1 = 0
        l2 = 0
        if self.average_across_node_types:
            summary_batch = positives.mean(dim=0)
            if self.convex_combination_weight is not None:
                if self.global_summary is not None:
                    convex_combination_weight = self.convex_combination_weight
                    self.global_summary = (convex_combination_weight *
                        summary_batch + (1 - convex_combination_weight) *
                        self.global_summary.detach())
                else:
                    self.global_summary = summary_batch
                summary_batch = self.global_summary
            summary = torch.sigmoid(summary_batch)
            positive = self.discriminator(positives.mean(dim=0), summary)
            negative = self.discriminator(negatives.mean(dim=0), summary)
            l1 += self.loss(positive, torch.ones_like(positive))
            l2 += self.loss(negative, torch.zeros_like(negative))
            return l1 + l2
        else:
            raise NotImplementedError


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_hidden': 4}]
