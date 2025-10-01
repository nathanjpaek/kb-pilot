import torch


class SvmProbsLoss(torch.nn.Module):

    def __init__(self):
        super(SvmProbsLoss, self).__init__()

    def forward(self, decisions, logits, targets, multi_label=False):
        y = targets.float()
        svm_targets = y * 2 - 1
        projection_dist = 1 - svm_targets * decisions
        margin = torch.max(torch.zeros_like(projection_dist), projection_dist)
        svm_loss = margin.mean()
        n_plus = torch.sum(y, dim=0)
        n_minus = torch.sum(1.0 - y, dim=0)
        n_plus_rate = (n_plus + 1.0) / (n_plus + 2.0)
        n_minus_rate = 1.0 / (n_minus + 2.0)
        y_cv = n_plus_rate * y + n_minus_rate * (1 - y)
        y_hat = torch.sigmoid(logits) if multi_label else torch.softmax(logits,
            dim=-1)
        platt_loss = -1 * torch.mean(y_cv * torch.log(y_hat) + (1 - y_cv) *
            torch.log(1 - y_hat))
        return svm_loss + platt_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
