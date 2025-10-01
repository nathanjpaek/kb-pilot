import torch


class WeightedBCELoss(torch.nn.Module):

    def __init__(self, neg_scale=-1, bce_sum=False):
        super(WeightedBCELoss, self).__init__()
        self.log_sigmoid = torch.nn.LogSigmoid()
        self.neg_scale = neg_scale
        self.bce_sum = bce_sum

    def forward(self, logits, targets, target_weights):
        neg_vals = self.log_sigmoid(-logits) * (1 - targets)
        if self.neg_scale > 0:
            neg_vals *= self.neg_scale
        vals = -targets * self.log_sigmoid(logits) - neg_vals
        if self.bce_sum:
            losses = torch.sum(vals * target_weights, dim=-1)
        else:
            losses = torch.sum(vals * target_weights, dim=-1) / logits.size()[1
                ]
        return torch.mean(losses)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
