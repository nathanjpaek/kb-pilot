import torch
import torch.multiprocessing
import torch.utils.data


class StandardNLL(torch.nn.modules.loss._Loss):
    """
    Shape:
        log_prob:   batch x time x class
        y_true:     batch x time
        mask:       batch x time
        output:     batch
    """

    def forward(self, log_prob, y_true, mask):
        mask = mask.float()
        log_P = torch.gather(log_prob.view(-1, log_prob.size(2)), 1, y_true
            .contiguous().view(-1, 1))
        log_P = log_P.view(y_true.size(0), y_true.size(1))
        log_P = log_P * mask
        sum_log_P = torch.sum(log_P, dim=1) / torch.sum(mask, dim=1)
        return -sum_log_P


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
