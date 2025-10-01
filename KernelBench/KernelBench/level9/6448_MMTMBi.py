import torch
import torch.nn as nn
from typing import Sequence


class MMTMBi(nn.Module):
    """
    bi moludal fusion
    """

    def __init__(self, dim_tab, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: feature dimension of tabular data
        dim_img: feature dimension of MIL image modal
        ratio
        """
        super(MMTMBi, self).__init__()
        dim = dim_tab + dim_img
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_tab = nn.Linear(dim_out, dim_tab)
        self.fc_img = nn.Linear(dim_out, dim_img)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tab_feat, img_feat) ->Sequence[torch.Tensor]:
        """

        Parameters
        ----------
        tab_feat: b * c
        skeleton: b * c

        Returns
            表格数据加权结果
            WSI 全局特征加权结果
            WSI 全局特征加权权重
        -------

        """
        squeeze = torch.cat([tab_feat, img_feat], dim=1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        tab_out = self.fc_tab(excitation)
        img_out = self.fc_img(excitation)
        tab_out = self.sigmoid(tab_out)
        img_out = self.sigmoid(img_out)
        return tab_feat * tab_out, img_feat * img_out, img_out


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_tab': 4, 'dim_img': 4}]
