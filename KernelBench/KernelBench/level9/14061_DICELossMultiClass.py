import torch
import torch.nn as nn


class DICELossMultiClass(nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, input_mask):
        num_classes = output.size(1) - 1
        dice_eso = 0
        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :], 1)
            mask = torch.squeeze(input_mask[:, i, :, :], 1)
            num = probs * mask
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)
            den1 = probs * probs
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)
            den2 = mask * mask
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)
            eps = 1e-07
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            dice_eso += dice
        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
