import torch
import torch.nn as nn


class OuterProductLayer(nn.Module):
    """OutterProduct Layer used in PNN. This implemention is
    adapted from code that the author of the paper published on https://github.com/Atomu2014/product-nets.
    """

    def __init__(self, num_feature_field, embedding_size, device):
        """
        Args:
            num_feature_field(int) :number of feature fields.
            embedding_size(int) :number of embedding size.
            device(torch.device) : device object of the model.
        """
        super(OuterProductLayer, self).__init__()
        self.num_feature_field = num_feature_field
        num_pairs = int(num_feature_field * (num_feature_field - 1) / 2)
        embed_size = embedding_size
        self.kernel = nn.Parameter(torch.rand(embed_size, num_pairs,
            embed_size), requires_grad=True)
        nn.init.xavier_uniform_(self.kernel)
        self

    def forward(self, feat_emb):
        """
        Args:
            feat_emb(torch.FloatTensor) :3D tensor with shape: [batch_size,num_pairs,embedding_size].

        Returns:
            outer_product(torch.FloatTensor): The outer product of input tensor. shape of [batch_size, num_pairs]
        """
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]
        q = feat_emb[:, col]
        p.unsqueeze_(dim=1)
        p = torch.mul(p, self.kernel.unsqueeze(0))
        p = torch.sum(p, dim=-1)
        p = torch.transpose(p, 2, 1)
        outer_product = p * q
        return outer_product.sum(dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_feature_field': 4, 'embedding_size': 4, 'device': 0}]
