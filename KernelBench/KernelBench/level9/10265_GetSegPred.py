import torch
import torch.utils.data.dataset


class GetSegPred(torch.nn.Module):

    def __init__(self, scale):
        super(GetSegPred, self).__init__()
        self.scale = scale // 2

    def forward(self, segs, ptcloud):
        temp_cloud = torch.round((ptcloud + 1) * self.scale - 0.501).long()
        temp_cloud[temp_cloud == -1] = 0
        segsT = torch.transpose(segs, 1, 4)
        preds = []
        for i, p in enumerate(temp_cloud):
            pred = segsT[i, p[:, 0], p[:, 1], p[:, 2]].unsqueeze(dim=0)
            preds.append(pred)
        return torch.cat(preds, dim=0).contiguous()


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
