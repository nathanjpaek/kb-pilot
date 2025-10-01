import torch
import torch.nn as nn
import torch.utils.data


class RLFeatPreprocessNet(nn.Module):
    """
    Preprocess Features
    1. visual feature
    2. label prediction embed feature
    3. box embed
    4. overlap embed
    """

    def __init__(self, feat_size, embed_size, bbox_size, overlap_size,
        output_size):
        super(RLFeatPreprocessNet, self).__init__()
        self.feature_size = feat_size
        self.embed_size = embed_size
        self.box_info_size = bbox_size
        self.overlap_info_size = overlap_size
        self.output_size = output_size
        self.resize_feat = nn.Linear(self.feature_size, int(output_size / 4))
        self.resize_embed = nn.Linear(self.embed_size, int(output_size / 4))
        self.resize_box = nn.Linear(self.box_info_size, int(output_size / 4))
        self.resize_overlap = nn.Linear(self.overlap_info_size, int(
            output_size / 4))
        self.resize_feat.weight.data.normal_(0, 0.001)
        self.resize_embed.weight.data.normal_(0, 0.01)
        self.resize_box.weight.data.normal_(0, 1)
        self.resize_overlap.weight.data.normal_(0, 1)
        self.resize_feat.bias.data.zero_()
        self.resize_embed.bias.data.zero_()
        self.resize_box.bias.data.zero_()
        self.resize_overlap.bias.data.zero_()

    def forward(self, obj_feat, obj_embed, box_info, overlap_info):
        resized_obj = self.resize_feat(obj_feat)
        resized_embed = self.resize_embed(obj_embed)
        resized_box = self.resize_box(box_info)
        resized_overlap = self.resize_overlap(overlap_info)
        output_feat = torch.cat((resized_obj, resized_embed, resized_box,
            resized_overlap), 1)
        return output_feat


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feat_size': 4, 'embed_size': 4, 'bbox_size': 4,
        'overlap_size': 4, 'output_size': 4}]
