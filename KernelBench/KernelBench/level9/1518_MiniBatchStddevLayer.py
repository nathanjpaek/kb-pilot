import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import torch.utils.cpp_extension


class AllGatherLayer(autograd.Function):
    """All gather layer with backward propagation path.

    Indeed, this module is to make ``dist.all_gather()`` in the backward graph.
    Such kind of operation has been widely used in Moco and other contrastive
    learning algorithms.
    """

    @staticmethod
    def forward(ctx, x):
        """Forward function."""
        ctx.save_for_backward(x)
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward function."""
        x, = ctx.saved_tensors
        grad_out = torch.zeros_like(x)
        grad_out = grad_outputs[dist.get_rank()]
        return grad_out


class MiniBatchStddevLayer(nn.Module):
    """Minibatch standard deviation.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        eps (float, optional):  Epsilon value to avoid computation error.
            Defaults to 1e-8.
        gather_all_batch (bool, optional): Whether gather batch from all GPUs.
            Defaults to False.
    """

    def __init__(self, group_size=4, eps=1e-08, gather_all_batch=False):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.gather_all_batch = gather_all_batch
        if self.gather_all_batch:
            assert torch.distributed.is_initialized(
                ), 'Only in distributed training can the tensors be all gathered.'

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.gather_all_batch:
            x = torch.cat(AllGatherLayer.apply(x), dim=0)
        assert x.shape[0] <= self.group_size or x.shape[0
            ] % self.group_size == 0, f'Batch size be smaller than or equal to group size. Otherwise, batch size should be divisible by the group size.But got batch size {x.shape[0]}, group size {self.group_size}'
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        y = torch.reshape(x, (group_size, -1, c, h, w))
        y = y - y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0, keepdim=False)
        y = torch.sqrt(y + self.eps)
        y = y.mean(dim=(1, 2, 3), keepdim=True)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
