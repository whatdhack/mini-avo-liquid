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
