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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
