import torch
import numpy as np

from .functions import bm_t
from .type import Tensor


def merge_leading_dims(tensor, ndims=1):
    assert ndims <= tensor.ndim
    shape = tensor.size()
    lead_dim_size = np.prod(shape[:ndims])
    tensor = tensor.view(lead_dim_size, *shape[ndims:])
    return tensor


def torch2numpy(tensor):
    return tensor.data.cpu().numpy()


def grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2
    )
    return total_norm


def prepend_time_zero(times: Tensor, target: Tensor, mask: Tensor):
    """Prepends t = 0 to the batch

    Args:
        times (Tensor): 1D Tensor of times without t = 0 as the first time
        target (Tensor): B x T x D Tensor
        mask (Tensor): Mask
    """
    B, _, D = target.size()
    times = torch.cat(
        [torch.zeros([1], device=times.device), times],
        dim=0,
    )
    target = torch.cat(
        [
            torch.zeros((B, 1, D), device=target.device),
            target,
        ],
        dim=1,
    )
    if mask.ndim == 3:
        mask = torch.cat([torch.zeros((B, 1, D), device=mask.device), mask], dim=1)
    else:
        mask = torch.cat([torch.zeros((B, 1), device=mask.device), mask], dim=1)
    return times, target, mask


def skew_symmetric_init_(tensor, gain=1):
    if tensor.ndim not in {2, 3}:
        raise ValueError("Only tensors with 2 or 3 dimensions are supported")

    rand_matrix = torch.rand_like(tensor)
    if tensor.ndim == 2:
        rand_matrix.unsqueeze_(0)

    skew_symm_matrix = (rand_matrix - bm_t(rand_matrix)) / 2

    with torch.no_grad():
        tensor.view_as(skew_symm_matrix).copy_(skew_symm_matrix)
        tensor.mul_(gain)
    return tensor
