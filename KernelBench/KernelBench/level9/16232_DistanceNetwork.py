import torch
import torch.nn as nn


class DistanceNetwork(nn.Module):

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_magnitude = sum_support.clamp(eps, float('inf')).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.
                unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
