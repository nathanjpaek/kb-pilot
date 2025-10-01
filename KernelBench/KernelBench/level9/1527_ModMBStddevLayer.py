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


class ModMBStddevLayer(nn.Module):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.

    Note that to accelerate the training procedure, we also add a new feature
    of ``sync_std`` to achieve multi-nodes/machine training. This feature is
    still in beta version and we have tested it on 256 scales.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        channel_groups (int, optional): The size of groups in channel
            dimension. Defaults to 1.
        sync_std (bool, optional): Whether to use synchronized std feature.
            Defaults to False.
        sync_groups (int | None, optional): The size of groups in node
            dimension. Defaults to None.
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self, group_size=4, channel_groups=1, sync_std=False,
        sync_groups=None, eps=1e-08):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_std = sync_std
        self.sync_groups = group_size if sync_groups is None else sync_groups
        if self.sync_std:
            assert torch.distributed.is_initialized(
                ), 'Only in distributed training can the sync_std be activated.'
            mmcv.print_log('Adopt synced minibatch stddev layer', 'mmgen')

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map with shape of (N, C+1, H, W).
        """
        if self.sync_std:
            all_features = torch.cat(AllGatherLayer.apply(x), dim=0)
            rank, ws = get_dist_info()
            local_bs = all_features.shape[0] // ws
            start_idx = local_bs * rank
            if start_idx + self.sync_groups > all_features.shape[0]:
                start_idx = all_features.shape[0] - self.sync_groups
            end_idx = min(local_bs * rank + self.sync_groups, all_features.
                shape[0])
            x = all_features[start_idx:end_idx]
        assert x.shape[0] <= self.group_size or x.shape[0
            ] % self.group_size == 0, f'Batch size be smaller than or equal to group size. Otherwise, batch size should be divisible by the group size.But got batch size {x.shape[0]}, group size {self.group_size}'
        assert x.shape[1
            ] % self.channel_groups == 0, f'"channel_groups" must be divided by the feature channels. channel_groups: {self.channel_groups}, feature channels: {x.shape[1]}'
        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        y = torch.reshape(x, (group_size, -1, self.channel_groups, c //
            self.channel_groups, h, w))
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + self.eps)
        y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
