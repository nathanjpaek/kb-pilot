import torch
import torch.nn as nn


class MyLoss(nn.Module):

    def __init__(self):
        super(MyLoss, self).__init__()
        None
        self.reduce_var = True
    pass
    """
    weights has shape (n), multiply loss of point i with weights[i]
    """

    def forward(self, outputs, y, weights, calculate_add=True):
        nns = torch.trunc(y)
        nns = nns.long()
        k = nns.shape[1]
        n = nns.shape[0]
        n = outputs.shape[0]
        batch_size = y.shape[0]
        n_bins = outputs.shape[1]
        diff = 0
        booster_weights = 0
        if calculate_add:
            reshaped_nns = torch.movedim(nns, 1, 0)
            del nns
            reshaped_nns = torch.unsqueeze(reshaped_nns, 2)
            reshaped_nns = torch.movedim(reshaped_nns, 1, 2)
            reshaped_nns = reshaped_nns.repeat(1, n_bins, 1)
            refactored_outputs = torch.unsqueeze(outputs, 0)
            refactored_outputs = torch.movedim(refactored_outputs, 1, 2)
            refactored_outputs = refactored_outputs.repeat(k, 1, 1)
            cost_tensor_new = torch.gather(refactored_outputs, 2, reshaped_nns)
            del reshaped_nns
            del refactored_outputs
            reshaped_outputs = torch.transpose(outputs, 0, 1)
            reshaped_outputs = torch.reshape(reshaped_outputs, (1, n_bins, n))
            reshaped_outputs = reshaped_outputs[:, :, :batch_size]
            add = cost_tensor_new + reshaped_outputs
            del reshaped_outputs
            del cost_tensor_new
            add, idx = torch.max(add, 1)
            del idx
            booster_weights = torch.mean(add, 0)
            booster_weights = 2 - booster_weights
            booster_weights = booster_weights / 2
            booster_weights = torch.clamp(booster_weights, min=0.5)
            add = add * weights
            add = torch.mean(add)
            diff = torch.square(2 - add)
        pass
        target_b = n / n_bins
        batch_outputs = outputs[:batch_size, :]
        b = torch.sum(batch_outputs, 0)
        b_max = torch.max(b)
        b_min = torch.min(b)
        b = b_max - b_min
        del batch_outputs
        cost = b / target_b + diff
        b = b.detach()
        diff = diff.detach()
        booster_weights = booster_weights.detach()
        return cost, diff, b / target_b, booster_weights
    pass


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
