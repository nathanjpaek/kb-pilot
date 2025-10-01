import torch
import torch.nn as nn


class NormAttnMap(nn.Module):

    def __init__(self, norm_type='cossim'):
        super(NormAttnMap, self).__init__()
        self.norm_type = norm_type

    def forward(self, attn_map):
        if self.norm_type != 'cosssim':
            norm = torch.max(attn_map, dim=1, keepdim=True)[0].detach()
        else:
            norm = torch.max(torch.abs(attn_map), dim=1, keepdim=True)[0
                ].detach()
        norm[norm <= 1] = 1
        attn = attn_map / norm
        return attn, norm


class MergeModule(nn.Module):

    def __init__(self, norm_type='cossim', need_norm=True):
        super(MergeModule, self).__init__()
        self.norm_type = norm_type
        self.need_norm = need_norm
        self.norm_fun = NormAttnMap(norm_type)

    def forward(self, attn_map, global_sub_attn_maps, global_obj_attn_maps,
        mask_sub, mask_obj):
        bs, num_seq, n = global_sub_attn_maps.size(0
            ), global_sub_attn_maps.size(1), global_sub_attn_maps.size(2)
        mask_sub_expand = (mask_sub == 1).float().unsqueeze(2).expand(bs,
            num_seq, n)
        sub_attn_map_sum = torch.sum(mask_sub_expand * global_sub_attn_maps,
            dim=1)
        mask_obj_expand = (mask_obj == 1).float().unsqueeze(2).expand(bs,
            num_seq, n)
        obj_attn_map_sum = torch.sum(mask_obj_expand * global_obj_attn_maps,
            dim=1)
        attn_map_sum = sub_attn_map_sum + obj_attn_map_sum + attn_map
        if self.need_norm:
            attn, _norm = self.norm_fun(attn_map_sum)
        else:
            attn = attn_map_sum
        return attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
