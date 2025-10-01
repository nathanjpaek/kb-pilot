import torch
import torch.nn.functional as F
import torch.nn as nn


def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-06).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    sum_centroids = sum_centroids.reshape(sum_centroids.shape[0], 1,
        sum_centroids.shape[-1])
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)
    utterance_centroids_flat = utterance_centroids.view(utterance_centroids
        .shape[0] * utterance_centroids.shape[1], -1)
    embeddings_flat = embeddings.view(embeddings.shape[0] * num_utterances, -1)
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[
        0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.
        shape[0], 1)
    embeddings_expand = embeddings_expand.view(embeddings_expand.shape[0] *
        embeddings_expand.shape[1], embeddings_expand.shape[-1])
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(embeddings.size(0), num_utterances, centroids.
        size(0))
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0],
        num_utterances)
    cos_diff = cos_diff + 1e-06
    return cos_diff


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-06)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0}]
