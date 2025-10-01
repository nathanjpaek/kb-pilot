import torch
import numpy as np


class Conditional_Contrastive_loss_plus(torch.nn.Module):

    def __init__(self, device, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss_plus, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, 'h and w should be same'
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.bool)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature,
        margin):
        p2i_similarity_matrix = self.calculate_similarity_matrix(proxy,
            inst_embed)
        i2i_similarity_matrix = self.calculate_similarity_matrix(inst_embed,
            inst_embed)
        p2i_similarity_zone = torch.exp((p2i_similarity_matrix - margin) /
            temperature)
        i2i_similarity_zone = torch.exp((i2i_similarity_matrix - margin) /
            temperature)
        mask_4_remove_negatives = negative_mask[labels]
        p2i_positives = p2i_similarity_zone * mask_4_remove_negatives
        i2i_positives = i2i_similarity_zone * mask_4_remove_negatives
        p2i_numerator = p2i_positives.sum(dim=1)
        i2i_numerator = i2i_positives.sum(dim=1)
        p2i_denomerator = p2i_similarity_zone.sum(dim=1)
        i2i_denomerator = i2i_similarity_zone.sum(dim=1)
        p2i_contra_loss = -torch.log(temperature * (p2i_numerator /
            p2i_denomerator)).mean()
        i2i_contra_loss = -torch.log(temperature * (i2i_numerator /
            i2i_denomerator)).mean()
        return p2i_contra_loss + i2i_contra_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.ones(
        [4], dtype=torch.int64), torch.ones([4], dtype=torch.int64), torch.
        rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0, 'batch_size': 4, 'pos_collected_numerator': 4}]
