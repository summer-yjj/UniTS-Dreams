import torch.distributed as dist
import torch
import os


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    # ========= 单卡 / 非分布式 =========
    if not dist.is_available() or not dist.is_initialized():
        # 如果不是 torchrun / 没有分布式环境变量
        if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
            args.distributed = False
            args.rank = 0
            args.world_size = 1
            print("Single GPU mode (DDP disabled)")
            return

    # ========= 分布式模式 =========

    dist.init_process_group(
        backend="nccl",
    )
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    print(f"Start running basic DDP on rank {rank}.")

    dist.barrier()
    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def gather_tensors_from_all_gpus(tensor_list, device_id, to_numpy=True):
    """
    Gather tensors from all GPUs in a DDP setup onto each GPU.

    Args:
    local_tensors (list of torch.Tensor): List of tensors on the local GPU.

    Returns:
    list of torch.Tensor: List of all tensors gathered from all GPUs, available on each GPU.
    """
    # 1. 检查是否为单卡环境
    if not dist.is_available() or not dist.is_initialized():
        # 如果是单卡，直接走一遍 to_numpy 逻辑后返回即可
        if to_numpy:
            return [t.cpu().numpy() if hasattr(t, 'cpu') else t for t in tensor_list]
        return tensor_list

    world_size = dist.get_world_size()
    tensor_list = [tensor.to(device_id).contiguous() for tensor in tensor_list]
    gathered_tensors = [[] for _ in range(len(tensor_list))]

    # Gathering tensors from all GPUs
    for tensor in tensor_list:
        # Each GPU will gather tensors from all other GPUs
        gathered_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_list, tensor)
        gathered_tensors.append(gathered_list)
    del tensor_list
    # Flattening the gathered list
    flattened_tensors = [
        tensor for sublist in gathered_tensors for tensor in sublist]
    del gathered_tensors
    if to_numpy:
        flattened_tensors_numpy = [tensor.cpu().numpy()
                                   for tensor in flattened_tensors]
        del flattened_tensors

        return flattened_tensors_numpy
    else:
        return flattened_tensors
