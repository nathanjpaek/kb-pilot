import torch
from torch import nn


class SeqAttendImgFusion(nn.Module):

    def __init__(self, seq_dim, img_dim, hidden_dim, att_dropout,
        att_scale_factor=1, **kwargs):
        super(SeqAttendImgFusion, self).__init__()
        self.SeqTrans = nn.Linear(seq_dim, hidden_dim, bias=False)
        self.ImgTrans = nn.Linear(img_dim, hidden_dim, bias=False)
        self.WeightTrans = nn.Linear(hidden_dim, 1, bias=False)
        self.AttentionScaleFactor = att_scale_factor
        if att_dropout is None:
            self.Dropout = nn.Identity()
        else:
            self.Dropout = nn.Dropout(att_dropout)

    def _attend(self, seq_features, img_features, res_att=False):
        assert len(img_features.size()
            ) == 3, f'[SeqAttendImgFusion] Input image shape must have 3 dimensions (batch, patch(w*h), feature(channel)), but got {img_features.size()}'
        img_feature_dim = img_features.size(-1)
        patch_size = img_features.size(1)
        seq_repeated_features = seq_features.unsqueeze(1).repeat((1,
            patch_size, 1))
        attend_output = torch.tanh(self.SeqTrans(seq_repeated_features)
            ) * torch.tanh(self.ImgTrans(img_features))
        attend_output = self.Dropout(attend_output)
        attend_alpha = torch.softmax(self.AttentionScaleFactor * self.
            WeightTrans(attend_output), dim=1)
        if res_att:
            res_ones = torch.ones_like(attend_alpha)
            attend_alpha = attend_alpha + res_ones
        attend_alpha = attend_alpha.repeat((1, 1, img_feature_dim))
        attended_img = torch.sum(img_features * attend_alpha, dim=1)
        return attended_img

    def forward(self, seq_features, img_features, **kwargs):
        raise NotImplementedError


class SeqAttendImgCatFusion(SeqAttendImgFusion):

    def __init__(self, seq_dim, img_dim, hidden_dim, att_dropout, **kwargs):
        super(SeqAttendImgCatFusion, self).__init__(seq_dim, img_dim,
            hidden_dim, att_dropout, **kwargs)

    def forward(self, seq_features, img_features, **kwargs):
        attended_img = self._attend(seq_features, img_features)
        return torch.cat((seq_features, attended_img), dim=-1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'seq_dim': 4, 'img_dim': 4, 'hidden_dim': 4, 'att_dropout':
        0.5}]
