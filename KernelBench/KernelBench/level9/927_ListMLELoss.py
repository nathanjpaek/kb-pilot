import torch
import torch.nn as nn


class ListMLELoss(nn.Module):

    def __init__(self):
        super(ListMLELoss, self).__init__()
        return

    def forward(self, y_pred, y_true, eps=1e-05, padded_value_indicator=-1):
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]
        _y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=
            indices)
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(
            dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + eps
            ) - preds_sorted_by_true_minus_max
        return torch.mean(torch.sum(observation_loss, dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
