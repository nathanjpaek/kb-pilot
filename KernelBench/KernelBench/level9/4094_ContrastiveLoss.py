from torch.nn import Module
import torch
from torch.nn import LogSoftmax
from torch.nn.functional import cosine_similarity


class ContrastiveLoss(Module):
    """A contrastive loss adapted from SimCLR.

    Link to SimCLR: https://arxiv.org/abs/2002.05709v3.
    """

    def __init__(self, temperature: 'float'=1.0):
        """Save hyper-params."""
        super().__init__()
        self.temperature = temperature
        self._log_softmax_fn = LogSoftmax(dim=-1)

    def forward(self, inputs: 'torch.Tensor', targets: 'torch.Tensor'
        ) ->torch.Tensor:
        """Get the loss."""
        inputs = inputs.permute(2, 3, 0, 1)
        targets = targets.permute(2, 3, 0, 1)
        batch_size = inputs.shape[-2]
        left = torch.cat([inputs, targets], -2).unsqueeze(-1)
        right = left.permute(0, 1, 4, 3, 2)
        similarity = cosine_similarity(left, right, dim=-2, eps=torch.finfo
            (left.dtype).eps)
        mask = torch.eye(2 * batch_size, device=similarity.device).bool()
        mask_nd = mask.unsqueeze(0).unsqueeze(0).tile(*similarity.shape[:2],
            1, 1)
        neg_inf = float('-inf') * torch.ones_like(similarity)
        similarity = torch.where(mask_nd, neg_inf, similarity)
        log_softmax = self._log_softmax_fn(similarity / self.temperature)
        positive_pairs = torch.cat([torch.diagonal(log_softmax, offset=
            batch_size, dim1=-2, dim2=-1), torch.diagonal(log_softmax,
            offset=-batch_size, dim1=-2, dim2=-1)], -1)
        return -positive_pairs.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
