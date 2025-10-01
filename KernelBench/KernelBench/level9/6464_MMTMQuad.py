import torch
import torch.nn as nn
from typing import Sequence


class MMTMQuad(nn.Module):
    """
    quad modal fusion
    """

    def __init__(self, dim_tab, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: feature dimension of tabular data
        dim_img: feature dimension of MIL model
        ratio
        """
        super(MMTMQuad, self).__init__()
        dim = dim_tab + dim_img * 3
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_tab = nn.Linear(dim_out, dim_tab)
        self.fc_img_scale1 = nn.Linear(dim_out, dim_img)
        self.fc_img_scale2 = nn.Linear(dim_out, dim_img)
        self.fc_img_scale3 = nn.Linear(dim_out, dim_img)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tab_feat, img_feat_scale1, img_feat_scale2,
        img_feat_scale3) ->Sequence[torch.Tensor]:
        """

        Parameters
        ----------
        tab_feat: b * c
        skeleton: b * c

        Returns
        -------

        """
        squeeze = torch.cat([tab_feat, img_feat_scale1, img_feat_scale2,
            img_feat_scale3], dim=1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        tab_out = self.fc_tab(excitation)
        img_out_scale1 = self.fc_img_scale1(excitation)
        img_out_scale2 = self.fc_img_scale2(excitation)
        img_out_scale3 = self.fc_img_scale3(excitation)
        tab_out = self.sigmoid(tab_out)
        img_out_scale1 = self.sigmoid(img_out_scale1)
        img_out_scale2 = self.sigmoid(img_out_scale2)
        img_out_scale3 = self.sigmoid(img_out_scale3)
        return (tab_feat * tab_out, img_feat_scale1 * img_out_scale1,
            img_out_scale1, img_feat_scale2 * img_out_scale2,
            img_out_scale2, img_feat_scale2 * img_out_scale3, img_out_scale3)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_tab': 4, 'dim_img': 4}]
