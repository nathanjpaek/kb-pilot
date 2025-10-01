import torch
from torch import nn
from torch import tensor


class NegativeSamplingLoss(nn.Module):
    """
    loss function of negative-sampling.

    """

    def forward(self, input_vectors: 'tensor', output_vectors: 'tensor',
        noise_vectors: 'tensor'):
        batch_size, embed_size = input_vectors.shape
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        torch.bmm(output_vectors, input_vectors)
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid(
            ).log()
        noise_loss = noise_loss.squeeze().sum(1)
        return -(out_loss + noise_loss).mean()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 1, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
