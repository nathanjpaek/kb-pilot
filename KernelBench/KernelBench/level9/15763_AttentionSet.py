import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, mode_dims, expand_dims, center_use_offset, att_type,
        bn, nat, name='Real'):
        super(Attention, self).__init__()
        self.center_use_offset = center_use_offset
        self.bn = bn
        self.nat = nat
        if center_use_offset:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims *
                2, mode_dims))
        else:
            self.atten_mats1 = nn.Parameter(torch.FloatTensor(expand_dims,
                mode_dims))
        nn.init.xavier_uniform(self.atten_mats1)
        self.register_parameter('atten_mats1_%s' % name, self.atten_mats1)
        if self.nat >= 2:
            self.atten_mats1_1 = nn.Parameter(torch.FloatTensor(mode_dims,
                mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_1)
            self.register_parameter('atten_mats1_1_%s' % name, self.
                atten_mats1_1)
        if self.nat >= 3:
            self.atten_mats1_2 = nn.Parameter(torch.FloatTensor(mode_dims,
                mode_dims))
            nn.init.xavier_uniform(self.atten_mats1_2)
            self.register_parameter('atten_mats1_2_%s' % name, self.
                atten_mats1_2)
        if bn != 'no':
            self.bn1 = nn.BatchNorm1d(mode_dims)
            self.bn1_1 = nn.BatchNorm1d(mode_dims)
            self.bn1_2 = nn.BatchNorm1d(mode_dims)
        if att_type == 'whole':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims, 1))
        elif att_type == 'ele':
            self.atten_mats2 = nn.Parameter(torch.FloatTensor(mode_dims,
                mode_dims))
        nn.init.xavier_uniform(self.atten_mats2)
        self.register_parameter('atten_mats2_%s' % name, self.atten_mats2)

    def forward(self, center_embed, offset_embed=None):
        if self.center_use_offset:
            temp1 = torch.cat([center_embed, offset_embed], dim=1)
        else:
            temp1 = center_embed
        if self.nat >= 1:
            if self.bn == 'no':
                temp2 = F.relu(temp1.mm(self.atten_mats1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1(temp1.mm(self.atten_mats1)))
            elif self.bn == 'after':
                temp2 = self.bn1(F.relu(temp1.mm(self.atten_mats1)))
        if self.nat >= 2:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_1))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_1(temp2.mm(self.atten_mats1_1)))
            elif self.bn == 'after':
                temp2 = self.bn1_1(F.relu(temp2.mm(self.atten_mats1_1)))
        if self.nat >= 3:
            if self.bn == 'no':
                temp2 = F.relu(temp2.mm(self.atten_mats1_2))
            elif self.bn == 'before':
                temp2 = F.relu(self.bn1_2(temp2.mm(self.atten_mats1_2)))
            elif self.bn == 'after':
                temp2 = self.bn1_2(F.relu(temp2.mm(self.atten_mats1_2)))
        temp3 = temp2.mm(self.atten_mats2)
        return temp3


class AttentionSet(nn.Module):

    def __init__(self, mode_dims, expand_dims, center_use_offset, att_reg=
        0.0, att_tem=1.0, att_type='whole', bn='no', nat=1, name='Real'):
        super(AttentionSet, self).__init__()
        self.center_use_offset = center_use_offset
        self.att_reg = att_reg
        self.att_type = att_type
        self.att_tem = att_tem
        self.Attention_module = Attention(mode_dims, expand_dims,
            center_use_offset, att_type=att_type, bn=bn, nat=nat)

    def forward(self, embeds1, embeds1_o, embeds2, embeds2_o, embeds3=[],
        embeds3_o=[]):
        temp1 = (self.Attention_module(embeds1, embeds1_o) + self.att_reg) / (
            self.att_tem + 0.0001)
        temp2 = (self.Attention_module(embeds2, embeds2_o) + self.att_reg) / (
            self.att_tem + 0.0001)
        if len(embeds3) > 0:
            temp3 = (self.Attention_module(embeds3, embeds3_o) + self.att_reg
                ) / (self.att_tem + 0.0001)
            if self.att_type == 'whole':
                combined = F.softmax(torch.cat([temp1, temp2, temp3], dim=1
                    ), dim=1)
                center = embeds1 * combined[:, 0].view(embeds1.size(0), 1
                    ) + embeds2 * combined[:, 1].view(embeds2.size(0), 1
                    ) + embeds3 * combined[:, 2].view(embeds3.size(0), 1)
            elif self.att_type == 'ele':
                combined = F.softmax(torch.stack([temp1, temp2, temp3]), dim=0)
                center = embeds1 * combined[0] + embeds2 * combined[1
                    ] + embeds3 * combined[2]
        elif self.att_type == 'whole':
            combined = F.softmax(torch.cat([temp1, temp2], dim=1), dim=1)
            center = embeds1 * combined[:, 0].view(embeds1.size(0), 1
                ) + embeds2 * combined[:, 1].view(embeds2.size(0), 1)
        elif self.att_type == 'ele':
            combined = F.softmax(torch.stack([temp1, temp2]), dim=0)
            center = embeds1 * combined[0] + embeds2 * combined[1]
        return center


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'mode_dims': 4, 'expand_dims': 4, 'center_use_offset': 4}]
