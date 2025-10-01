import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F
import torch.utils.model_zoo


def pdist(A, squared=False, eps=0.0001):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    if squared:
        return res
    else:
        return res.clamp(min=eps).sqrt()


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.
    Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = data.min(dim, keepdim=True)[0]
    masked_maximums = torch.max(torch.mul(data - axis_minimums, mask), dim=
        dim, keepdim=True)[0] + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
    Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = data.max(dim, keepdim=True)[0]
    masked_minimums = torch.min(torch.mul(data - axis_maximums, mask), dim=
        dim, keepdim=True)[0] + axis_maximums
    return masked_minimums


class TripletSemihardLoss(torch.nn.Module):
    """
    Computes the triplet loss with semi-hard negative mining.
    For every positive pair find only *one* hardest semihard negative example.

    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance - margin.
    The negative distance is selected among the neg pairsi
      which are at least greater than the positive distance (called semi-hard negatives),
      but withing the margin radius from anchor.
     If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.
    Args:
        labels: 1-D int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
    Returns:
        triplet_loss: float32 scalar.
    """

    def __init__(self, margin=1.0, soft=False, **kwargs):
        self.margin = margin
        self.soft = soft
        torch.nn.Module.__init__(self)

    def forward(self, embeddings, labels):
        d = pdist(embeddings)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]
            ).type_as(d) - torch.eye(len(d)).type_as(d)
        T = d.unsqueeze(1).expand(*((len(d),) * 3))
        M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
        dist_diff = T - T.transpose(1, 2)
        mask = M * (dist_diff < 0).type_as(M)
        mask_final = (M * (dist_diff < 0).type_as(M)).sum(dim=1, keepdim=True)
        mask_final = mask_final > 0
        mask_final.squeeze_(dim=1)
        assert len(mask_final.shape) == 2
        dist_diff_negatives_outside = masked_maximum(dist_diff, mask, dim=1
            ).squeeze_(dim=1)
        dist_diff_negatives_inside = masked_minimum(dist_diff, M, dim=1
            ).squeeze_(dim=1)
        dist_diff_semi_hard_negatives = torch.where(mask_final,
            dist_diff_negatives_outside, dist_diff_negatives_inside)
        if self.soft:
            loss_mat = dist_diff_semi_hard_negatives.exp().log1p()
        else:
            loss_mat = dist_diff_semi_hard_negatives + self.margin
        assert len(loss_mat.shape) == 2
        assert len(pos.shape) == 2
        return F.relu(pos * loss_mat).sum() / pos.sum()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4])]


def get_init_inputs():
    return [[], {}]
