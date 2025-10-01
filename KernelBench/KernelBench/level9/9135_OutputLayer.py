import torch
import torch.nn as nn
import torch.utils.dlpack


class OutputLayer(nn.Module):

    def __init__(self, voxel_size=1.0):
        super(OutputLayer, self).__init__()

    def forward(self, features_list, index_map_list):
        out = []
        for feat, index_map in zip(features_list, index_map_list):
            out.append(feat[index_map])
        return torch.cat(out, 0)


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype
        =torch.int64)]


def get_init_inputs():
    return [[], {}]
