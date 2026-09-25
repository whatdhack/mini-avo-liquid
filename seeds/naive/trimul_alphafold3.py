import torch
import triton
import triton.language as tl

@triton.jit
def trimul_naive(left_ptr, right_ptr, out_ptr, B, N, C, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Reference naive triangular outer-product
    pass
