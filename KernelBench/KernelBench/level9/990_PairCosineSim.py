import torch
import torch.utils.data
import torch.nn as nn


class PairCosineSim(nn.Module):

    def __init__(self):
        super(PairCosineSim, self).__init__()

    def forward(self, supports, target):
        """
        Calculates pairwise cosine similarity of support sets with target sample.

        :param supports: The embeddings of the support set samples, tensor of shape [batch_size, sequence_length, input_size]
        :param targets: The embedding of the target sample, tensor of shape [batch_size, input_size] -> [batch_size, sequence_length, input_size]

        :return: Tensor with cosine similarities of shape [batch_size, target_size, support_size]
        """
        eps = 1e-10
        similarities = []
        for support_image in supports:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_magnitude = sum_support.clamp(eps, float('inf')).rsqrt()
            target_unsqueeze = target.unsqueeze(1)
            support_image_unsqueeze = support_image.unsqueeze(2)
            dot_product = target_unsqueeze.bmm(support_image_unsqueeze)
            dot_product = dot_product.squeeze()
            cos_sim = dot_product * support_magnitude
            similarities.append(cos_sim)
        similarities = torch.stack(similarities)
        return similarities


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
