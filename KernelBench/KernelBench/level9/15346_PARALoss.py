import torch
import torch.nn as nn
import torch.nn.functional as F


class PARALoss(nn.Module):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()

    def forward(self, score, predicate_one_hot_labels):
        entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True
            ).repeat_interleave(score.shape[1], dim=1)
        entity_mask = (entity_mask > 0).float()
        entity_sum = (entity_mask != 0).sum(dim=(2, 3)).float()
        loss = ((F.binary_cross_entropy(score, predicate_one_hot_labels,
            reduction='none') * entity_mask).sum(dim=(2, 3)) / entity_sum
            ).mean()
        if loss.item() < 0:
            None
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
