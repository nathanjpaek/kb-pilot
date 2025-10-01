import torch
import torch.optim


class KL(torch.nn.KLDivLoss):

    def __init__(self, is_input_log: 'bool'=False, is_target_log: 'bool'=False
        ):
        super(KL, self).__init__(reduction='none', log_target=is_target_log)
        self.is_input_log = is_input_log

    def forward(self, gt: 'torch.Tensor', pred: 'torch.Tensor') ->torch.Tensor:
        return super(KL, self).forward(pred if self.is_input_log else pred.
            log(), gt)


class Lambda(KL):

    def __init__(self, lamda: 'float'=0.5, is_input_log: 'bool'=False,
        is_target_log: 'bool'=False, epsilon: 'float'=1e-24):
        super(Lambda, self).__init__(True, True)
        self.lamda = lamda
        self.is_input_log_ = is_input_log
        self.is_target_log_ = is_target_log
        self.epsilon = epsilon

    def forward(self, gt: 'torch.Tensor', pred: 'torch.Tensor') ->torch.Tensor:
        m = self.lamda * (pred.exp() if self.is_input_log_ else pred) + (
            1.0 - self.lamda) * (gt.exp() if self.is_target_log_ else gt)
        m = m.log()
        p = pred if self.is_input_log_ else (pred + self.epsilon).log()
        g = gt if self.is_target_log_ else (gt + self.epsilon).log()
        pred_to_m = super(Lambda, self).forward(p, m)
        gt_to_m = super(Lambda, self).forward(g, m)
        lamda_divergence = self.lamda * pred_to_m + (1.0 - self.lamda
            ) * gt_to_m
        return lamda_divergence


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
