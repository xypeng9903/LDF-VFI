# Modified from:
# https://github.com/Wan-Video/Wan2.2/blob/388807310646ed5f318a99f8e8d9ad28c5b65373/wan/distributed/util.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as F


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_rank():
    return dist.get_rank()


def get_world_size(group=None):
    return dist.get_world_size(group)


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size(group)
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        F.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor, group=None):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = F.all_gather(tensor, group=group)
    return tensor_list


def gather_forward(input, dim, group=None):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input, group)
    return torch.cat(output, dim=dim).contiguous()