import torch
import torch.nn as nn


class CodeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, origin_logit, trans_logit):
        origin_code, trans_code = torch.sign(origin_logit), torch.sign(
            trans_logit)
        code_balance_loss = (torch.mean(torch.abs(torch.sum(origin_code,
            dim=1))) + torch.mean(torch.abs(torch.sum(trans_code, dim=1)))) / 2
        code_loss = self.loss(trans_code, origin_code.detach())
        return code_balance_loss, code_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
