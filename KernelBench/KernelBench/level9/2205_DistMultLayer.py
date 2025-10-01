import torch
import torch.utils.data
import torch.nn as nn


class DistMultLayer(nn.Module):

    def __init__(self):
        super(DistMultLayer, self).__init__()

    def forward(self, sub_emb, obj_emb, rel_emb):
        return torch.sum(sub_emb * obj_emb * rel_emb, dim=-1)

    def predict(self, sub_emb, obj_emb, rel_emb):
        return torch.matmul(sub_emb * rel_emb, obj_emb.t())


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
