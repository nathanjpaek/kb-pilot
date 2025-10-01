import torch
import torch.nn as nn


class DistillLoss(nn.Module):

    def __init__(self):
        super(DistillLoss, self).__init__()

    def forward(self, t_feat, feat, cams):
        assert len(cams) == feat.shape[0] and feat.shape[0] == t_feat.shape[0]
        t_feat = t_feat / t_feat.norm(p=2, dim=1, keepdim=True)
        t_dist = self.cdist(t_feat, t_feat)
        feat = feat / feat.norm(p=2, dim=1, keepdim=True)
        dist = self.cdist(feat, feat)
        same_cam_mask = torch.eq(cams.unsqueeze(1), cams.unsqueeze(0)).float()
        for i in range(len(same_cam_mask)):
            same_cam_mask[i, i] = 0
        same_cam_mask = same_cam_mask if cams.is_cuda else same_cam_mask
        diff = (t_dist - dist) * same_cam_mask
        mse_loss = torch.norm(diff) / feat.shape[0]
        return mse_loss

    def cdist(self, a, b):
        """
        Returns euclidean distance between (all feature pairs) in a and b

        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff ** 2).sum(2) + 1e-12).sqrt()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
