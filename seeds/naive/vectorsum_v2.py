import torch
import triton
import triton.language as tl

@triton.jit
def vectorsum_naive(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x, axis=0)
    tl.atomic_add(output_ptr, sum_val)


def solve(x: torch.Tensor) -> torch.Tensor:
    """Reference launch: one 1024-element block per program, atomics into a scalar."""
    out = torch.zeros(1, device=x.device, dtype=torch.float32)
    n = x.numel()
    BLOCK_SIZE = 1024
    vectorsum_naive[(triton.cdiv(n, BLOCK_SIZE),)](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out
