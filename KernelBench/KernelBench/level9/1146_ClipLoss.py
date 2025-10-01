import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.distributed.nn


def gather_features(image_features, text_features, aug1_embed=None,
    aug2_embed=None, local_loss=False, gather_with_grad=False, rank=0,
    world_size=1, horovod=False):
    if horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
            if aug1_embed is not None and aug2_embed is not None:
                all_aug1_embed = hvd.allgather(aug1_embed)
                all_aug2_embed = hvd.allgather(aug2_embed)
            else:
                all_aug1_embed, all_aug2_embed = None, None
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
                if aug1_embed is not None and aug2_embed is not None:
                    all_aug1_embed = hvd.allgather(aug1_embed)
                    all_aug2_embed = hvd.allgather(aug2_embed)
                else:
                    all_aug1_embed, all_aug2_embed = None, None
            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(
                    world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(
                    world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                if aug1_embed is not None and aug2_embed is not None:
                    gathered_aug1_embed = list(all_aug1_embed.chunk(
                        world_size, dim=0))
                    gathered_aug2_embed = list(all_aug2_embed.chunk(
                        world_size, dim=0))
                    gathered_aug1_embed[rank] = aug1_embed
                    gathered_aug2_embed[rank] = aug2_embed
                    all_aug1_embed = torch.cat(gathered_aug1_embed, dim=0)
                    all_aug2_embed = torch.cat(gathered_aug2_embed, dim=0)
                else:
                    all_aug1_embed, all_aug2_embed = None, None
    elif gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(
            image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(
            text_features), dim=0)
        if aug1_embed is not None and aug2_embed is not None:
            all_aug1_embed = torch.cat(torch.distributed.nn.all_gather(
                aug1_embed), dim=0)
            all_aug2_embed = torch.cat(torch.distributed.nn.all_gather(
                aug2_embed), dim=0)
        else:
            all_aug1_embed, all_aug2_embed = None, None
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in
            range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in
            range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if aug1_embed is not None and aug2_embed is not None:
            gathered_aug1_embed = [torch.zeros_like(aug1_embed) for _ in
                range(world_size)]
            gathered_aug2_embed = [torch.zeros_like(aug2_embed) for _ in
                range(world_size)]
            dist.all_gather(gathered_aug1_embed, aug1_embed)
            dist.all_gather(gathered_aug2_embed, aug2_embed)
            all_aug1_embed = torch.cat(gathered_aug1_embed, dim=0)
            all_aug2_embed = torch.cat(gathered_aug2_embed, dim=0)
            if not local_loss:
                all_aug1_embed[rank] = aug1_embed
                all_aug2_embed[rank] = aug2_embed
        else:
            all_aug1_embed, all_aug2_embed = None, None
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    return (all_image_features, all_text_features, all_aug1_embed,
        all_aug2_embed)


class ClipLoss(nn.Module):

    def __init__(self, local_loss=False, gather_with_grad=False, rank=0,
        world_size=1, horovod=False):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.horovod = horovod
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features, _, _ = gather_features(
                image_features, text_features, None, None, self.local_loss,
                self.gather_with_grad, self.rank, self.world_size, self.horovod
                )
            if self.local_loss:
                logits_per_image = (logit_scale * image_features @
                    all_text_features.T)
                logits_per_text = (logit_scale * text_features @
                    all_image_features.T)
            else:
                logits_per_image = (logit_scale * all_image_features @
                    all_text_features.T)
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            self.labels[device] = labels
        else:
            labels = self.labels[device]
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.
            cross_entropy(logits_per_text, labels)) / 2
        return total_loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
