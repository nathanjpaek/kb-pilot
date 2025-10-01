import torch


class TorchJaccardLoss(torch.nn.modules.Module):

    def __init__(self):
        super(TorchJaccardLoss, self).__init__()

    def forward(self, outputs, targets):
        eps = 1e-15
        jaccard_target = (targets == 1).float()
        jaccard_output = torch.sigmoid(outputs)
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        jaccard_score = (intersection + eps) / (union - intersection + eps)
        self._stash_jaccard = jaccard_score
        loss = 1.0 - jaccard_score
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
