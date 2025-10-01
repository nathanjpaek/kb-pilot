import torch
import torch.nn.functional as F
import torch.onnx


class L1RankLoss(torch.nn.Module):
    """
    L1 loss + Rank loss
    """

    def __init__(self, **kwargs):
        super(L1RankLoss, self).__init__()
        self.l1_w = kwargs.get('l1_w', 1)
        self.rank_w = kwargs.get('rank_w', 1)
        self.hard_thred = kwargs.get('hard_thred', 1)
        self.use_margin = kwargs.get('use_margin', False)

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        l1_loss = F.l1_loss(preds, gts) * self.l1_w
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (
            torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label -
                img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(-masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w
        return loss_total


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
