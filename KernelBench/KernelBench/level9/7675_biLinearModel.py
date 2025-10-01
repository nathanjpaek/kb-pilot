import torch
import torch.distributed
import torch
import torch.nn as nn


class biLinearModel(nn.Module):
    """Currently just for a pair"""

    def __init__(self, hidden_size):
        super(biLinearModel, self).__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, doc_emb, group_embs, candi_sent_masks):
        """
        doc_emb: batch_size, 1, emb_dim
        group_emb: batch_size, max_sent_count, emb_dim
        candi_sent_masks: batch_size, max_group_count
        """
        doc_emb = doc_emb.expand_as(group_embs)
        h_0 = self.bilinear(group_embs.contiguous(), doc_emb.contiguous())
        sent_group_scores = h_0.squeeze(-1) * candi_sent_masks.float()
        return sent_group_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
