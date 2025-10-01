import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch._utils
import torch.optim


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def __init__(self, weight=None, reduction='mean', pos_weight=None,
        ignore_all_zeros=False):
        if pos_weight is not None:
            if isinstance(pos_weight, str):
                pos_weight_path = pos_weight
                with open(pos_weight_path) as weights_file:
                    weights_dict = json.load(weights_file)
                num_classes = len(weights_dict)
                pos_weight = torch.ones([num_classes])
                for k, v in weights_dict.items():
                    pos_weight[int(k)] = v
                None
            elif isinstance(pos_weight, list):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float)
                None
        super().__init__(weight=weight, reduction=reduction, pos_weight=
            pos_weight)
        self.ignore_all_zeros = ignore_all_zeros

    def forward(self, input, target):
        if self.ignore_all_zeros and target.ndim == 4:
            non_zeros = target.sum(dim=1) > 0
            target = target[non_zeros]
            input = input[non_zeros]
        return F.binary_cross_entropy_with_logits(input, target.float(),
            self.weight, pos_weight=self.pos_weight, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
