import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.utils.data.distributed


class MultiSoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self, class_weight=None, label_smoothing_value=0):
        super(MultiSoftmaxCrossEntropyLoss, self).__init__()
        self.class_weight = class_weight
        if self.class_weight is not None:
            self.class_weight = self.class_weight
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.label_smoothing_value = label_smoothing_value

    def forward(self, input, target):
        return self.cross_entropy(input, target, self.class_weight)

    def cross_entropy(self, pred, soft_targets, class_weight=None):
        if class_weight is not None:
            class_weight_matrix = class_weight.expand_as(soft_targets)
            used_class_weights = th.where(soft_targets > 0,
                class_weight_matrix, soft_targets)
            samples_weight = th.max(used_class_weights, dim=1, keepdim=True)[0]
            loss = th.mean(th.sum(-samples_weight * soft_targets * self.
                logsoftmax(pred), 1), 0)
        else:
            if self.label_smoothing_value > 0:
                batch_size, total_classes_count = soft_targets.size()
                for sample_index in range(batch_size):
                    pos_indices = np.where(soft_targets[sample_index, :] > 0)
                    pos_classes_count = len(pos_indices[0])
                    if pos_classes_count > 0:
                        neg_p = self.label_smoothing_value / float(
                            total_classes_count - pos_classes_count)
                        pos_p = self.label_smoothing_value / pos_classes_count
                        soft_targets[sample_index, :] += neg_p
                        soft_targets[sample_index, pos_indices[0]
                            ] = soft_targets[sample_index, pos_indices[0]
                            ] - pos_p - neg_p
            loss = th.sum(-soft_targets * self.logsoftmax(pred))
            loss = loss / soft_targets.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
