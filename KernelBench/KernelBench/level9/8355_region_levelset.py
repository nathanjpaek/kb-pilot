import torch
import torch.nn as nn


class region_levelset(nn.Module):
    """
    the mian of leveset function
    """

    def __init__(self):
        super(region_levelset, self).__init__()

    def forward(self, mask_score, norm_img, class_weight):
        """
        mask_score: predcited mask scores tensor:(N,C,W,H)
        norm_img: normalizated images tensor:(N,C,W,H)
        class_weight: weight for different classes
        """
        mask_score_shape = mask_score.shape
        norm_img_shape = norm_img.shape
        level_set_loss = 0.0
        for i in range(norm_img_shape[1]):
            norm_img_ = torch.unsqueeze(norm_img[:, i], 1)
            norm_img_ = norm_img_.expand(norm_img_shape[0],
                mask_score_shape[1], norm_img_shape[2], norm_img_shape[3])
            ave_similarity = torch.sum(norm_img_ * mask_score, (2, 3)
                ) / torch.sum(mask_score, (2, 3))
            ave_similarity = ave_similarity.view(norm_img_shape[0],
                mask_score_shape[1], 1, 1)
            region_level = norm_img_ - ave_similarity.expand(norm_img_shape
                [0], mask_score_shape[1], norm_img_shape[2], norm_img_shape[3])
            region_level_loss = (class_weight * region_level * region_level *
                mask_score)
            level_set_loss += torch.sum(region_level_loss)
        return level_set_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
