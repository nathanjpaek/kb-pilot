import torch
import torch.nn as nn
import torch.nn.functional as F


class PARALossSoftmax(nn.Module):
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
        soft = True
        if predicate_one_hot_labels.is_sparse:
            predicate_one_hot_labels = predicate_one_hot_labels.to_dense()
        if not soft:
            entity_mask = predicate_one_hot_labels.sum(dim=1)
            label = predicate_one_hot_labels.argmax(dim=1)
            loss = F.cross_entropy(score, label, reduction='none')
            loss = loss * entity_mask
            loss = loss.sum(dim=(1, 2)) / entity_mask.sum(dim=(1, 2))
            loss = loss.mean()
        else:
            entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True
                ).repeat_interleave(score.shape[1], dim=1).float()
            score = (score * entity_mask).sum(dim=(2, 3))
            label = predicate_one_hot_labels.sum(dim=(2, 3)).argmax(dim=-1)
            loss = F.cross_entropy(score, label)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
