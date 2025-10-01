import torch


class PartialBCELoss(torch.nn.Module):

    def __init__(self):
        super(PartialBCELoss, self).__init__()
        self.log_sigmoid = torch.nn.LogSigmoid()

    def forward(self, logits, targets, targets_mask, weights=None):
        pos_vals = -targets * self.log_sigmoid(logits)
        neg_vals = -self.log_sigmoid(-logits) * (1 - targets)
        vals = pos_vals + neg_vals
        losses = torch.sum(vals * targets_mask, dim=-1)
        if weights is not None:
            losses *= weights
        return torch.mean(losses)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
