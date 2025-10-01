import torch
import torch.nn as nn


class InnerProductLayer(nn.Module):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.

    """

    def __init__(self, num_feature_field, device):
        """
        Args:
            num_feature_field(int) :number of feature fields.
            device(torch.device) : device object of the model.
        """
        super(InnerProductLayer, self).__init__()
        self.num_feature_field = num_feature_field
        self

    def forward(self, feat_emb):
        """
        Args:
            feat_emb(torch.FloatTensor) :3D tensor with shape: [batch_size,num_pairs,embedding_size].

        Returns:
            inner_product(torch.FloatTensor): The inner product of input tensor. shape of [batch_size, num_pairs]
        """
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]
        q = feat_emb[:, col]
        inner_product = p * q
        return inner_product.sum(dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_feature_field': 4, 'device': 0}]
