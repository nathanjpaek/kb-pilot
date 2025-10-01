import torch


class Homoscedastic(torch.nn.Module):
    """https://arxiv.homoscedasticorg/abs/1705.07115"""

    def __init__(self, n_tasks, reduction='sum'):
        super(Homoscedastic, self).__init__()
        self.n_tasks = n_tasks
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        device = losses.device
        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device)
        coeffs = 1 / stds ** 2
        multi_task_losses = coeffs * losses + torch.log(stds)
        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()
        return multi_task_losses


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_tasks': 4}]
