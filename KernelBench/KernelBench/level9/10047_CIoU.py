import torch
from torch import nn


class CIoU(nn.Module):

    def __init__(self):
        super(CIoU, self).__init__()

    def forward(self, inputs, targets):
        size = len(inputs)
        uL_truth = targets[:, 0:2]
        lR_truth = targets[:, 2:4]
        uL_pred = inputs[:, 0:2]
        lR_pred = inputs[:, 2:4]
        truth_cen = torch.div(torch.add(uL_truth, lR_truth), 2)
        pred_cen = torch.div(torch.add(uL_pred, lR_pred), 2)
        uL_truth_x = uL_truth[:, 0]
        uL_truth_y = uL_truth[:, 1]
        lR_truth_x = lR_truth[:, 0]
        lR_truth_y = lR_truth[:, 1]
        uL_pred_x = uL_pred[:, 0]
        uL_pred_y = uL_pred[:, 1]
        lR_pred_x = lR_pred[:, 0]
        lR_pred_y = lR_pred[:, 1]
        truth_cen_x = truth_cen[:, 0]
        truth_cen_y = truth_cen[:, 1]
        pred_cen_x = pred_cen[:, 0]
        pred_cen_y = pred_cen[:, 1]
        p = torch.sqrt((pred_cen_x - truth_cen_x) ** 2 + (pred_cen_y -
            truth_cen_y) ** 2)
        enc_X = torch.reshape(torch.minimum(uL_truth, uL_pred), (size, 2))
        enc_Y = torch.reshape(torch.maximum(lR_truth, lR_pred), (size, 2))
        bounding_box = torch.reshape(torch.cat((enc_X, enc_Y), 1), (size, 4))
        bb_uL_x = bounding_box[:, 0]
        bb_uL_y = bounding_box[:, 1]
        bb_lR_x = bounding_box[:, 2]
        bb_lR_y = bounding_box[:, 3]
        (bb_lR_x - bb_uL_x) * (bb_lR_y - bb_uL_y)
        C = torch.sqrt((bb_lR_x - bb_uL_x) ** 2 + (bb_lR_y - bb_uL_y) ** 2)
        X = torch.where(torch.gt(uL_pred_x, lR_truth_x) | torch.gt(
            uL_truth_x, lR_pred_x), 0, 1) * (torch.minimum(lR_truth_x,
            lR_pred_x) - torch.maximum(uL_truth_x, uL_pred_x))
        Y = torch.where(torch.gt(uL_pred_y, lR_truth_y) | torch.gt(
            uL_truth_y, lR_pred_y), 0, 1) * (torch.minimum(lR_truth_y,
            lR_pred_y) - torch.maximum(uL_truth_y, uL_pred_y))
        i_area = X * Y
        rec_1 = (lR_truth_x - uL_truth_x) * (lR_truth_y - uL_truth_y)
        rec_2 = (lR_pred_x - uL_pred_x) * (lR_pred_y - uL_pred_y)
        total_area = rec_1 + rec_2 - i_area
        IoU = i_area / total_area
        DIoU = 1 - IoU + p ** 2 / C ** 2
        return torch.mean(DIoU)
        """#my own calculation
        first = 1 - (i_area/BBA) # this will approach 0 when i_area == bbA
        second = torch.where(p<5, first, torch.mul(p, first))
        return torch.mean(second)"""
        """pred_W = lR_pred_x - uL_pred_x
        pred_H = torch.where((lR_pred_y - uL_pred_y)!=0, 1, 0)*(lR_pred_y - uL_pred_y)

        truth_W = lR_truth_x - uL_truth_x
        truth_H = torch.where((lR_truth_y - uL_truth_y)!=0, 1, 0)*(lR_truth_y - uL_truth_y)

        
        V = (4/(np.pi**2))*((torch.atan(torch.div(truth_W, truth_H)) - torch.atan(torch.div(pred_W, pred_H)))**2)

        #alpha_1 = torch.div(V, ((1 - IoU)+V))
        #print("alpha 1: ", alpha_1)

        alpha = torch.where(torch.lt(IoU, 0.5), 0, 1)*torch.div(V, ((1-IoU)+V))

        #print(torch.where(torch.lt(IoU, 0.5), 0, 1))

        CIoU = 1 - (IoU - (((p**2)/(C**2)) + alpha*V))
        return CIoU"""


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
