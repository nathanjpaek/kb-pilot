import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self.
            _embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 /
            self._num_embeddings)
        self._commitment_cost = commitment_cost
        self.emb_indexes = []

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = torch.sum(flat_input ** 2, dim=1, keepdim=True
            ) + torch.sum(self._embedding.weight ** 2, dim=1
            ) - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        self.emb_indexes.extend(encoding_indices.cpu().detach().numpy()[0])
        encodings = torch.zeros(encoding_indices.shape[0], self.
            _num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs +
            1e-10)))
        return loss, quantized.permute(0, 3, 1, 2).contiguous(
            ), perplexity, encodings


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_embeddings': 4, 'embedding_dim': 4, 'commitment_cost': 4}
        ]
