import torch


def torch_cdf_loss(tensor_a, tensor_b, p=1):
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)
    if p == 1:
        cdf_distance = torch.sum(torch.abs(cdf_tensor_a - cdf_tensor_b), dim=-1
            )
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow(cdf_tensor_a -
            cdf_tensor_b, 2), dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a -
            cdf_tensor_b), p), dim=-1), 1 / p)
    cdf_loss = cdf_distance.mean()
    return cdf_loss


def torch_wasserstein_loss(tensor_a, tensor_b):
    return torch_cdf_loss(tensor_a, tensor_b, p=1)


class WassersteinLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tensor_a, tensor_b):
        return torch_wasserstein_loss(tensor_a, tensor_b)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
